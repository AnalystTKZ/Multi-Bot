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
2026-05-01 09:19:50,993 INFO Loading feature-engineered data...
2026-05-01 09:19:51,696 INFO Loaded 221743 rows, 202 features
2026-05-01 09:19:51,697 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-05-01 09:19:51,700 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-05-01 09:19:51,700 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-05-01 09:19:51,701 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-05-01 09:19:51,701 INFO No leakage confirmed: train < val < test timestamps

=== SPLIT COMPLETE (CALENDAR, no shuffling) ===
  Train:      130,951 bars  2016-01-04 → 2021-08-05
  Validation:  44,000 bars  2021-08-05 → 2023-08-04  ← Round 1 backtest
  Test:        46,792 bars  2023-08-07 → 2025-08-05  ← Blind / Round 2 backtest
  Features:   202
  Leakage check: PASS
  DONE  Step 5 - Split

  Data split (calendar):
    train         130951 bars  2016-01-04 → 2021-08-05
    validation     44000 bars  2021-08-05 → 2023-08-04
    test           46792 bars  2023-08-07 → 2025-08-05

=== Phase 7a: Train GRU + Regime (train set only) ===
  START Step 7a - GRU+Regime
2026-05-01 09:19:54,448 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-01 09:19:54,448 INFO --- Training regime ---
2026-05-01 09:19:54,449 INFO Running retrain --model regime
2026-05-01 09:19:54,660 INFO retrain environment: KAGGLE
2026-05-01 09:19:56,608 INFO Device: CUDA (2 GPU(s))
2026-05-01 09:19:56,619 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 09:19:56,620 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 09:19:56,620 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-01 09:19:56,621 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-01 09:19:56,622 INFO Retrain data split: train
2026-05-01 09:19:56,623 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-05-01 09:19:56,823 INFO NumExpr defaulting to 4 threads.
2026-05-01 09:19:57,070 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-01 09:19:57,070 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 09:19:57,070 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 09:19:57,070 INFO Regime phase macro_correlations: 0.0s
2026-05-01 09:19:57,071 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-01 09:19:57,144 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-05-01 09:19:57,189 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-05-01 09:19:57,190 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:19:57,209 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:19:57,227 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:19:57,246 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:19:57,266 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:19:57,284 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:19:57,303 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:19:57,321 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:19:57,340 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:19:57,358 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:19:57,381 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:19:57,521 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:57,571 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:57,593 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:57,593 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:57,603 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:57,604 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:19:58,045 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1756, 'BIAS_DOWN': 1771, 'BIAS_NEUTRAL': 4875}  ambiguous=13 (total=8402) horizon=12
2026-05-01 09:19:58,047 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-05-01 09:19:58,235 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:58,279 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:58,299 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:58,300 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:58,309 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:58,310 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:19:58,711 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1846, 'BIAS_DOWN': 1663, 'BIAS_NEUTRAL': 4893}  ambiguous=13 (total=8402) horizon=12
2026-05-01 09:19:58,713 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-05-01 09:19:58,890 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:58,930 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:58,952 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:58,953 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:58,961 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:58,963 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:19:59,349 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1725, 'BIAS_DOWN': 1818, 'BIAS_NEUTRAL': 4859}  ambiguous=13 (total=8402) horizon=12
2026-05-01 09:19:59,350 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-05-01 09:19:59,536 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:59,578 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:59,599 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:59,600 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:59,608 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:19:59,609 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:19:59,996 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1852, 'BIAS_DOWN': 1677, 'BIAS_NEUTRAL': 4873}  ambiguous=13 (total=8402) horizon=12
2026-05-01 09:19:59,998 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-05-01 09:20:00,176 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:00,219 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:00,241 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:00,242 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:00,250 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:00,251 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:00,710 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1717, 'BIAS_DOWN': 1858, 'BIAS_NEUTRAL': 4828}  ambiguous=13 (total=8403) horizon=12
2026-05-01 09:20:00,712 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-05-01 09:20:00,899 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:00,938 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:00,960 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:00,961 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:00,969 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:00,971 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:01,383 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1823, 'BIAS_DOWN': 1695, 'BIAS_NEUTRAL': 4885}  ambiguous=13 (total=8403) horizon=12
2026-05-01 09:20:01,385 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-05-01 09:20:01,540 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:20:01,574 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:20:01,595 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:20:01,595 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:20:01,603 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:20:01,604 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:02,009 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1837, 'BIAS_DOWN': 1733, 'BIAS_NEUTRAL': 4832}  ambiguous=13 (total=8402) horizon=12
2026-05-01 09:20:02,010 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-05-01 09:20:02,184 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:02,225 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:02,247 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:02,247 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:02,256 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:02,257 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:02,653 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1769, 'BIAS_DOWN': 1779, 'BIAS_NEUTRAL': 4854}  ambiguous=13 (total=8402) horizon=12
2026-05-01 09:20:02,655 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-05-01 09:20:02,833 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:02,873 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:02,894 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:02,895 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:02,904 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:02,905 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:03,305 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1642, 'BIAS_DOWN': 1875, 'BIAS_NEUTRAL': 4885}  ambiguous=13 (total=8402) horizon=12
2026-05-01 09:20:03,306 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-05-01 09:20:03,483 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:03,525 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:03,548 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:03,548 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:03,558 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:03,559 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:03,953 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1651, 'BIAS_DOWN': 1907, 'BIAS_NEUTRAL': 4845}  ambiguous=13 (total=8403) horizon=12
2026-05-01 09:20:03,955 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-05-01 09:20:04,243 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:20:04,308 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:20:04,335 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:20:04,336 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:20:04,348 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:20:04,350 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:20:05,179 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 4279, 'BIAS_DOWN': 4164, 'BIAS_NEUTRAL': 11374}  ambiguous=13 (total=19817) horizon=12
2026-05-01 09:20:05,181 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-05-01 09:20:05,349 INFO Regime phase HTF dataset build: 8.2s (103290 samples)
2026-05-01 09:20:05,350 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260501_092005
2026-05-01 09:20:05,654 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-05-01 09:20:05,656 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=103158 dropped=132 classes={'BIAS_UP': 21809, 'BIAS_DOWN': 21800, 'BIAS_NEUTRAL': 59549})
2026-05-01 09:20:05,685 INFO RegimeClassifier[mode=htf_bias]: 103158 samples, classes={'BIAS_UP': 21809, 'BIAS_DOWN': 21800, 'BIAS_NEUTRAL': 59549}, device=cuda
2026-05-01 09:20:05,686 INFO RegimeClassifier: sample weights — mean=0.787  ambiguous(<0.4)=0.0%
2026-05-01 09:20:05,686 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-05-01 09:20:05,686 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-01 09:20:11,139 INFO Regime epoch  1/50 — tr=1.1660 va=1.1090 acc=0.365 per_class={'BIAS_UP': 0.386, 'BIAS_DOWN': 0.237, 'BIAS_NEUTRAL': 0.403}
2026-05-01 09:20:11,347 INFO Regime epoch  2/50 — tr=1.1649 va=1.1089 acc=0.368
2026-05-01 09:20:11,544 INFO Regime epoch  3/50 — tr=1.1646 va=1.1089 acc=0.368
2026-05-01 09:20:11,749 INFO Regime epoch  4/50 — tr=1.1656 va=1.1088 acc=0.366
2026-05-01 09:20:11,959 INFO Regime epoch  5/50 — tr=1.1653 va=1.1091 acc=0.355 per_class={'BIAS_UP': 0.391, 'BIAS_DOWN': 0.252, 'BIAS_NEUTRAL': 0.38}
2026-05-01 09:20:12,159 INFO Regime epoch  6/50 — tr=1.1649 va=1.1089 acc=0.362
2026-05-01 09:20:12,363 INFO Regime epoch  7/50 — tr=1.1657 va=1.1089 acc=0.365
2026-05-01 09:20:12,561 INFO Regime epoch  8/50 — tr=1.1648 va=1.1090 acc=0.359
2026-05-01 09:20:12,764 INFO Regime epoch  9/50 — tr=1.1645 va=1.1090 acc=0.361
2026-05-01 09:20:12,988 INFO Regime epoch 10/50 — tr=1.1647 va=1.1092 acc=0.356 per_class={'BIAS_UP': 0.377, 'BIAS_DOWN': 0.256, 'BIAS_NEUTRAL': 0.384}
2026-05-01 09:20:13,192 INFO Regime epoch 11/50 — tr=1.1645 va=1.1092 acc=0.357
2026-05-01 09:20:13,399 INFO Regime epoch 12/50 — tr=1.1646 va=1.1092 acc=0.356
2026-05-01 09:20:13,591 INFO Regime epoch 13/50 — tr=1.1639 va=1.1093 acc=0.359
2026-05-01 09:20:13,787 INFO Regime epoch 14/50 — tr=1.1640 va=1.1093 acc=0.354
2026-05-01 09:20:13,787 INFO Regime early stop at epoch 14 (no_improve=10)
2026-05-01 09:20:13,805 WARNING RegimeClassifier accuracy 0.366 < warning floor 0.483 (harder structural labels; check blind backtest economics)
2026-05-01 09:20:13,810 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-01 09:20:13,810 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-01 09:20:13,811 INFO Regime phase HTF train: 8.2s
2026-05-01 09:20:13,937 INFO Regime HTF complete: acc=0.366, n=103290 per_class={'BIAS_UP': 0.381, 'BIAS_DOWN': 0.239, 'BIAS_NEUTRAL': 0.407}
2026-05-01 09:20:13,939 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:20:13,975 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 4279, 'BIAS_DOWN': 4164, 'BIAS_NEUTRAL': 11374}  ambiguous=13 (total=19817) horizon=12
2026-05-01 09:20:13,984 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 5.34875, 'BIAS_DOWN': 5.407792207792208, 'BIAS_NEUTRAL': 7.433986928104575}
2026-05-01 09:20:13,990 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4279, 'mean': 0.0017182910764822602, 'mean_over_std': 0.3954951269266541}, 'BIAS_DOWN': {'n': 4164, 'mean': -0.001754885052740925, 'mean_over_std': -0.3973602040268371}, 'BIAS_NEUTRAL': {'n': 11373, 'mean': 6.863221972092698e-05, 'mean_over_std': 0.021025233297328444}}
2026-05-01 09:20:13,991 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 4279, 'mean': 0.0017182910764822602, 'mean_over_std': 0.3954951269266541}, 'BIAS_DOWN': {'n': 4164, 'mean': -0.001754885052740925, 'mean_over_std': -0.3973602040268371}, 'BIAS_NEUTRAL': {'n': 11361, 'mean': 6.848051916093716e-05, 'mean_over_std': 0.020969591280060183}}
2026-05-01 09:20:13,994 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-05-01 09:20:13,997 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:13,998 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:14,000 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:14,002 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:14,004 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:14,006 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:14,007 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:14,009 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:14,011 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:14,012 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:14,015 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:20:14,028 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:14,030 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:14,031 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:14,031 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:14,032 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:14,034 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:14,573 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8942, 'RANGING': 13380, 'CONSOLIDATING': 8163, 'VOLATILE': 2253}  ambiguous=13 (total=32738) horizon=12
2026-05-01 09:20:14,576 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-05-01 09:20:14,709 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:14,711 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:14,712 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:14,713 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:14,713 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:14,716 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:15,220 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 7969, 'RANGING': 13888, 'CONSOLIDATING': 8178, 'VOLATILE': 2703}  ambiguous=13 (total=32738) horizon=12
2026-05-01 09:20:15,224 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-05-01 09:20:15,361 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:15,364 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:15,365 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:15,365 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:15,365 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:15,368 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:15,871 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8757, 'RANGING': 13514, 'CONSOLIDATING': 8167, 'VOLATILE': 2302}  ambiguous=13 (total=32740) horizon=12
2026-05-01 09:20:15,875 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-05-01 09:20:16,009 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:16,011 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:16,012 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:16,013 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:16,013 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:16,015 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:16,515 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8274, 'RANGING': 13643, 'CONSOLIDATING': 8178, 'VOLATILE': 2644}  ambiguous=13 (total=32739) horizon=12
2026-05-01 09:20:16,518 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-05-01 09:20:16,664 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:16,666 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:16,667 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:16,667 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:16,668 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:16,670 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:17,168 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8713, 'RANGING': 13564, 'CONSOLIDATING': 8168, 'VOLATILE': 2295}  ambiguous=13 (total=32740) horizon=12
2026-05-01 09:20:17,172 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-05-01 09:20:17,306 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:17,309 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:17,310 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:17,310 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:17,311 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:17,313 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:17,816 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8295, 'RANGING': 13768, 'CONSOLIDATING': 8178, 'VOLATILE': 2498}  ambiguous=13 (total=32739) horizon=12
2026-05-01 09:20:17,820 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-05-01 09:20:17,954 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:20:17,956 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:20:17,957 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:20:17,957 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:20:17,957 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:20:17,959 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:18,472 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 9045, 'RANGING': 13346, 'CONSOLIDATING': 8165, 'VOLATILE': 2183}  ambiguous=13 (total=32739) horizon=12
2026-05-01 09:20:18,475 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-05-01 09:20:18,610 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:18,612 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:18,613 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:18,613 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:18,614 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:18,616 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:19,119 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8133, 'RANGING': 13841, 'CONSOLIDATING': 8176, 'VOLATILE': 2590}  ambiguous=13 (total=32740) horizon=12
2026-05-01 09:20:19,122 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-05-01 09:20:19,255 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:19,257 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:19,258 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:19,259 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:19,259 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:19,261 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:19,770 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 7969, 'RANGING': 13883, 'CONSOLIDATING': 8177, 'VOLATILE': 2712}  ambiguous=13 (total=32741) horizon=12
2026-05-01 09:20:19,773 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-05-01 09:20:19,906 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:19,908 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:19,909 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:19,910 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:19,910 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:19,912 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:20,407 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8927, 'RANGING': 13395, 'CONSOLIDATING': 8160, 'VOLATILE': 2261}  ambiguous=13 (total=32743) horizon=12
2026-05-01 09:20:20,411 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-05-01 09:20:20,556 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:20:20,560 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:20:20,561 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:20:20,562 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:20:20,562 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:20:20,566 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:20:21,638 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 19458, 'RANGING': 31056, 'CONSOLIDATING': 18647, 'VOLATILE': 5463}  ambiguous=13 (total=74624) horizon=12
2026-05-01 09:20:21,644 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-05-01 09:20:21,947 INFO Regime phase LTF dataset build: 8.0s (401471 samples)
2026-05-01 09:20:21,948 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260501_092021
2026-05-01 09:20:21,953 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-05-01 09:20:21,957 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=401339 dropped=132 classes={'TRENDING': 104312, 'RANGING': 166920, 'CONSOLIDATING': 100239, 'VOLATILE': 29868})
2026-05-01 09:20:22,065 INFO RegimeClassifier[mode=ltf_behaviour]: 401339 samples, classes={'TRENDING': 104312, 'RANGING': 166920, 'CONSOLIDATING': 100239, 'VOLATILE': 29868}, device=cuda
2026-05-01 09:20:22,066 INFO RegimeClassifier: sample weights — mean=0.790  ambiguous(<0.4)=0.0%
2026-05-01 09:20:22,066 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-05-01 09:20:22,066 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-01 09:20:22,920 INFO Regime epoch  1/50 — tr=1.2597 va=1.2108 acc=0.374 per_class={'TRENDING': 0.192, 'RANGING': 0.246, 'CONSOLIDATING': 0.701, 'VOLATILE': 0.641}
2026-05-01 09:20:23,668 INFO Regime epoch  2/50 — tr=1.2597 va=1.2109 acc=0.376
2026-05-01 09:20:24,439 INFO Regime epoch  3/50 — tr=1.2602 va=1.2111 acc=0.376
2026-05-01 09:20:25,192 INFO Regime epoch  4/50 — tr=1.2591 va=1.2111 acc=0.375
2026-05-01 09:20:25,975 INFO Regime epoch  5/50 — tr=1.2593 va=1.2105 acc=0.375 per_class={'TRENDING': 0.197, 'RANGING': 0.244, 'CONSOLIDATING': 0.702, 'VOLATILE': 0.636}
2026-05-01 09:20:26,749 INFO Regime epoch  6/50 — tr=1.2589 va=1.2104 acc=0.373
2026-05-01 09:20:27,451 INFO Regime epoch  7/50 — tr=1.2599 va=1.2108 acc=0.377
2026-05-01 09:20:28,156 INFO Regime epoch  8/50 — tr=1.2594 va=1.2114 acc=0.377
2026-05-01 09:20:28,860 INFO Regime epoch  9/50 — tr=1.2591 va=1.2109 acc=0.379
2026-05-01 09:20:29,633 INFO Regime epoch 10/50 — tr=1.2587 va=1.2095 acc=0.373 per_class={'TRENDING': 0.203, 'RANGING': 0.239, 'CONSOLIDATING': 0.694, 'VOLATILE': 0.645}
2026-05-01 09:20:30,343 INFO Regime epoch 11/50 — tr=1.2586 va=1.2107 acc=0.375
2026-05-01 09:20:31,045 INFO Regime epoch 12/50 — tr=1.2589 va=1.2100 acc=0.376
2026-05-01 09:20:31,789 INFO Regime epoch 13/50 — tr=1.2587 va=1.2102 acc=0.376
2026-05-01 09:20:32,520 INFO Regime epoch 14/50 — tr=1.2584 va=1.2102 acc=0.376
2026-05-01 09:20:33,318 INFO Regime epoch 15/50 — tr=1.2586 va=1.2105 acc=0.377 per_class={'TRENDING': 0.204, 'RANGING': 0.248, 'CONSOLIDATING': 0.698, 'VOLATILE': 0.627}
2026-05-01 09:20:34,038 INFO Regime epoch 16/50 — tr=1.2589 va=1.2105 acc=0.378
2026-05-01 09:20:34,784 INFO Regime epoch 17/50 — tr=1.2586 va=1.2096 acc=0.371
2026-05-01 09:20:35,506 INFO Regime epoch 18/50 — tr=1.2582 va=1.2096 acc=0.375
2026-05-01 09:20:36,239 INFO Regime epoch 19/50 — tr=1.2582 va=1.2110 acc=0.379
2026-05-01 09:20:37,051 INFO Regime epoch 20/50 — tr=1.2590 va=1.2102 acc=0.375 per_class={'TRENDING': 0.212, 'RANGING': 0.243, 'CONSOLIDATING': 0.692, 'VOLATILE': 0.632}
2026-05-01 09:20:37,051 INFO Regime early stop at epoch 20 (no_improve=10)
2026-05-01 09:20:37,105 WARNING RegimeClassifier accuracy 0.373 < warning floor 0.400 (harder structural labels; check blind backtest economics)
2026-05-01 09:20:37,109 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-01 09:20:37,109 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-01 09:20:37,113 INFO Regime phase LTF train: 15.2s
2026-05-01 09:20:37,243 INFO Regime LTF complete: acc=0.373, n=401471 per_class={'TRENDING': 0.203, 'RANGING': 0.239, 'CONSOLIDATING': 0.694, 'VOLATILE': 0.645}
2026-05-01 09:20:37,247 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:20:37,346 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 19458, 'RANGING': 31056, 'CONSOLIDATING': 18647, 'VOLATILE': 5463}  ambiguous=13 (total=74624) horizon=12
2026-05-01 09:20:37,352 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 4.125079499682001, 'RANGING': 4.718322698268004, 'CONSOLIDATING': 5.990041760359782, 'VOLATILE': 3.723926380368098}
2026-05-01 09:20:37,362 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 19458, 'mean': 5.273978151244844e-05, 'mean_over_std': 0.021533559889529843}, 'RANGING': {'n': 31055, 'mean': -2.3091416023246664e-05, 'mean_over_std': -0.012142968527728414}, 'CONSOLIDATING': {'n': 18647, 'mean': 6.38696766285663e-05, 'mean_over_std': 0.041981339266451534}, 'VOLATILE': {'n': 5463, 'mean': -0.00012433839470805868, 'mean_over_std': -0.04867063159870624}}
2026-05-01 09:20:37,362 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 19458, 'mean': 5.273978151244844e-05, 'mean_over_std': 0.021533559889529843}, 'RANGING': {'n': 31043, 'mean': -2.2961083650504332e-05, 'mean_over_std': -0.01207348194887135}, 'CONSOLIDATING': {'n': 18647, 'mean': 6.38696766285663e-05, 'mean_over_std': 0.041981339266451534}, 'VOLATILE': {'n': 5463, 'mean': -0.00012433839470805868, 'mean_over_std': -0.04867063159870624}}
2026-05-01 09:20:37,366 INFO Regime retrain total: 40.7s (504761 samples)
2026-05-01 09:20:37,385 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-01 09:20:37,385 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 09:20:37,385 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 09:20:37,385 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-05-01 09:20:37,386 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-05-01 09:20:37,386 INFO Retrain complete. Total wall-clock: 40.8s
2026-05-01 09:20:40,113 INFO Model regime: SUCCESS
2026-05-01 09:20:40,113 INFO --- Training gru ---
2026-05-01 09:20:40,113 INFO Running retrain --model gru
2026-05-01 09:20:40,383 INFO retrain environment: KAGGLE
2026-05-01 09:20:42,249 INFO Device: CUDA (2 GPU(s))
2026-05-01 09:20:42,260 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 09:20:42,261 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 09:20:42,261 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-01 09:20:42,261 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-01 09:20:42,261 INFO Retrain data split: train
2026-05-01 09:20:42,262 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-05-01 09:20:42,428 INFO NumExpr defaulting to 4 threads.
2026-05-01 09:20:42,662 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-01 09:20:42,662 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 09:20:42,662 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 09:20:42,663 INFO GRU phase macro_correlations: 0.0s
2026-05-01 09:20:42,663 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-05-01 09:20:42,663 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260501_092042
2026-05-01 09:20:42,666 WARNING WeightsManifest: no manifest at /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json — assuming compatible (run a full retrain to generate one)
2026-05-01 09:20:42,835 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:42,858 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:42,874 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:42,882 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:42,884 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-01 09:20:42,884 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 09:20:42,884 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 09:20:42,885 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-05-01 09:20:42,886 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:42,993 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 17, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 8360}  ambiguous=8312 (total=8402)  short_runs_zeroed=15
2026-05-01 09:20:42,995 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:43,286 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 619, 'RANGING': 24879, 'CONSOLIDATING': 1882, 'VOLATILE': 5358}  ambiguous=26508 (total=32738)  short_runs_zeroed=2571
2026-05-01 09:20:43,317 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:43,642 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:43,799 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:43,924 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:44,165 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:44,188 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:44,204 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:44,214 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:44,215 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:44,314 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 31, 'BIAS_NEUTRAL': 8271}  ambiguous=8272 (total=8402)  short_runs_zeroed=52
2026-05-01 09:20:44,316 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:44,605 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 752, 'RANGING': 24932, 'CONSOLIDATING': 2099, 'VOLATILE': 4955}  ambiguous=26224 (total=32738)  short_runs_zeroed=2112
2026-05-01 09:20:44,625 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:44,949 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:45,104 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:45,225 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:45,454 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:45,477 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:45,494 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:45,502 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:45,503 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:45,605 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 46, 'BIAS_DOWN': 4, 'BIAS_NEUTRAL': 8352}  ambiguous=8316 (total=8402)  short_runs_zeroed=28
2026-05-01 09:20:45,607 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:45,895 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 617, 'RANGING': 24738, 'CONSOLIDATING': 2103, 'VOLATILE': 5282}  ambiguous=26101 (total=32740)  short_runs_zeroed=2218
2026-05-01 09:20:45,911 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:46,228 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:46,409 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:46,534 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:46,785 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:46,808 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:46,827 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:46,836 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:46,837 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:46,942 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 15, 'BIAS_DOWN': 19, 'BIAS_NEUTRAL': 8368}  ambiguous=8331 (total=8402)  short_runs_zeroed=27
2026-05-01 09:20:46,944 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:47,235 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 842, 'RANGING': 25174, 'CONSOLIDATING': 1996, 'VOLATILE': 4727}  ambiguous=26490 (total=32739)  short_runs_zeroed=2045
2026-05-01 09:20:47,262 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:47,582 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:47,738 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:47,862 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:48,085 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:48,109 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:48,126 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:48,135 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:48,136 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:48,240 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 93, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 8252}  ambiguous=8287 (total=8403)  short_runs_zeroed=75
2026-05-01 09:20:48,242 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:48,534 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 761, 'RANGING': 24579, 'CONSOLIDATING': 2089, 'VOLATILE': 5311}  ambiguous=25954 (total=32740)  short_runs_zeroed=2172
2026-05-01 09:20:48,550 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:48,871 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:49,026 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:49,143 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:49,367 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:49,388 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:49,405 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:49,413 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:49,414 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:49,515 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 87, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 8261}  ambiguous=8278 (total=8403)  short_runs_zeroed=63
2026-05-01 09:20:49,518 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:49,810 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 763, 'RANGING': 25215, 'CONSOLIDATING': 1968, 'VOLATILE': 4793}  ambiguous=26501 (total=32739)  short_runs_zeroed=1959
2026-05-01 09:20:49,830 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:50,143 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:50,298 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:50,418 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:50,622 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:20:50,643 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:20:50,659 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:20:50,667 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:20:50,667 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:50,768 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 21, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 8314}  ambiguous=8303 (total=8402)  short_runs_zeroed=39
2026-05-01 09:20:50,770 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:51,063 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 474, 'RANGING': 25095, 'CONSOLIDATING': 1967, 'VOLATILE': 5203}  ambiguous=26572 (total=32739)  short_runs_zeroed=2307
2026-05-01 09:20:51,077 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:51,411 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:51,567 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:51,686 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:51,907 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:51,928 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:51,944 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:51,953 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:51,954 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:52,054 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 37, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 8327}  ambiguous=8321 (total=8402)  short_runs_zeroed=44
2026-05-01 09:20:52,056 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:52,348 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 881, 'RANGING': 25133, 'CONSOLIDATING': 1987, 'VOLATILE': 4739}  ambiguous=26356 (total=32740)  short_runs_zeroed=1985
2026-05-01 09:20:52,368 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:52,694 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:52,854 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:52,974 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:53,196 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:53,218 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:53,235 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:53,245 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:53,246 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:53,345 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 8290}  ambiguous=8328 (total=8402)  short_runs_zeroed=61
2026-05-01 09:20:53,347 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:53,645 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 699, 'RANGING': 24989, 'CONSOLIDATING': 2130, 'VOLATILE': 4923}  ambiguous=26228 (total=32741)  short_runs_zeroed=2004
2026-05-01 09:20:53,664 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:54,002 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:54,161 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:54,285 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:54,508 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:54,530 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:54,548 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:54,556 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:20:54,557 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:54,658 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 42, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 8317}  ambiguous=8318 (total=8403)  short_runs_zeroed=44
2026-05-01 09:20:54,661 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:54,958 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 808, 'RANGING': 24290, 'CONSOLIDATING': 2180, 'VOLATILE': 5465}  ambiguous=25645 (total=32743)  short_runs_zeroed=2211
2026-05-01 09:20:54,977 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:55,305 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:55,465 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:55,585 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:20:55,928 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:20:55,957 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:20:55,975 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:20:55,987 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:20:55,988 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:20:56,181 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 107, 'BIAS_DOWN': 115, 'BIAS_NEUTRAL': 19595}  ambiguous=19636 (total=19817)  short_runs_zeroed=112
2026-05-01 09:20:56,184 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:20:56,852 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 1778, 'RANGING': 57285, 'CONSOLIDATING': 4542, 'VOLATILE': 11019}  ambiguous=60314 (total=74624)  short_runs_zeroed=4774
2026-05-01 09:20:56,903 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:20:57,580 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:20:57,830 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:20:57,996 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:20:58,134 INFO train_multi: 44 segments, ~6069667 total bars
2026-05-01 09:20:58,416 INFO GRULSTMPredictor: DataParallel across 2 GPUs ['Tesla T4', 'Tesla T4']
2026-05-01 09:20:58,416 INFO GRULSTMPredictor: model built (PyTorch, device=cuda)
2026-05-01 09:20:58,417 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-05-01 09:20:58,417 INFO train_multi: building combined dataset for TF=ALL (44 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
2026-05-01 09:22:01,222 INFO train_multi TF=ALL: 6068347 sequences across 44 segments
2026-05-01 09:22:01,223 INFO train_multi TF=ALL: estimated peak RAM = 11231 MB (train=479964 val=120008 n_feat=78 seq_len=30)
2026-05-01 09:22:02,719 INFO train_multi TF=ALL: train=479964 val=120008 (5623 MB tensors)
2026-05-01 09:22:10,096 INFO train_multi TF=ALL: cold-start — using OneCycleLR (max_lr=3e-04, patience=18, min_epochs=22)
2026-05-01 09:22:28,740 INFO train_multi TF=ALL epoch 1/50 train=0.8596 val=0.8528
2026-05-01 09:22:28,749 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:22:28,749 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:22:28,749 INFO train_multi TF=ALL: new best val=0.8528 — saved
2026-05-01 09:22:44,938 INFO train_multi TF=ALL epoch 2/50 train=0.8377 val=0.7944
2026-05-01 09:22:44,944 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:22:44,944 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:22:44,945 INFO train_multi TF=ALL: new best val=0.7944 — saved
2026-05-01 09:23:01,153 INFO train_multi TF=ALL epoch 3/50 train=0.7112 val=0.6879
2026-05-01 09:23:01,159 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:23:01,159 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:23:01,159 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-05-01 09:23:16,836 INFO train_multi TF=ALL epoch 4/50 train=0.6907 val=0.6879
2026-05-01 09:23:16,842 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:23:16,842 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:23:16,842 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-05-01 09:23:32,849 INFO train_multi TF=ALL epoch 5/50 train=0.6897 val=0.6879
2026-05-01 09:23:32,856 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:23:32,856 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:23:32,856 INFO train_multi TF=ALL: new best val=0.6879 — saved
2026-05-01 09:23:48,727 INFO train_multi TF=ALL epoch 6/50 train=0.6891 val=0.6881
2026-05-01 09:24:04,594 INFO train_multi TF=ALL epoch 7/50 train=0.6889 val=0.6879
2026-05-01 09:24:20,274 INFO train_multi TF=ALL epoch 8/50 train=0.6888 val=0.6880
2026-05-01 09:24:36,070 INFO train_multi TF=ALL epoch 9/50 train=0.6885 val=0.6878
2026-05-01 09:24:36,077 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:24:36,077 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:24:36,077 INFO train_multi TF=ALL: new best val=0.6878 — saved
2026-05-01 09:24:52,064 INFO train_multi TF=ALL epoch 10/50 train=0.6885 val=0.6877
2026-05-01 09:24:52,070 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:24:52,070 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:24:52,070 INFO train_multi TF=ALL: new best val=0.6877 — saved
2026-05-01 09:25:08,071 INFO train_multi TF=ALL epoch 11/50 train=0.6880 val=0.6883
2026-05-01 09:25:23,619 INFO train_multi TF=ALL epoch 12/50 train=0.6867 val=0.6876
2026-05-01 09:25:23,625 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:25:23,625 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:25:23,625 INFO train_multi TF=ALL: new best val=0.6876 — saved
2026-05-01 09:25:39,304 INFO train_multi TF=ALL epoch 13/50 train=0.6851 val=0.6848
2026-05-01 09:25:39,310 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:25:39,310 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:25:39,310 INFO train_multi TF=ALL: new best val=0.6848 — saved
2026-05-01 09:25:55,408 INFO train_multi TF=ALL epoch 14/50 train=0.6818 val=0.6791
2026-05-01 09:25:55,414 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:25:55,415 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:25:55,415 INFO train_multi TF=ALL: new best val=0.6791 — saved
2026-05-01 09:26:11,464 INFO train_multi TF=ALL epoch 15/50 train=0.6734 val=0.6692
2026-05-01 09:26:11,470 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:26:11,470 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:26:11,470 INFO train_multi TF=ALL: new best val=0.6692 — saved
2026-05-01 09:26:27,212 INFO train_multi TF=ALL epoch 16/50 train=0.6613 val=0.6504
2026-05-01 09:26:27,218 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:26:27,218 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:26:27,218 INFO train_multi TF=ALL: new best val=0.6504 — saved
2026-05-01 09:26:42,355 INFO train_multi TF=ALL epoch 17/50 train=0.6501 val=0.6404
2026-05-01 09:26:42,361 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:26:42,361 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:26:42,361 INFO train_multi TF=ALL: new best val=0.6404 — saved
2026-05-01 09:26:58,090 INFO train_multi TF=ALL epoch 18/50 train=0.6412 val=0.6305
2026-05-01 09:26:58,096 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:26:58,096 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:26:58,096 INFO train_multi TF=ALL: new best val=0.6305 — saved
2026-05-01 09:27:13,900 INFO train_multi TF=ALL epoch 19/50 train=0.6340 val=0.6407
2026-05-01 09:27:29,876 INFO train_multi TF=ALL epoch 20/50 train=0.6295 val=0.6231
2026-05-01 09:27:29,882 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:27:29,882 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:27:29,882 INFO train_multi TF=ALL: new best val=0.6231 — saved
2026-05-01 09:27:46,128 INFO train_multi TF=ALL epoch 21/50 train=0.6246 val=0.6195
2026-05-01 09:27:46,135 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:27:46,135 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:27:46,135 INFO train_multi TF=ALL: new best val=0.6195 — saved
2026-05-01 09:28:02,517 INFO train_multi TF=ALL epoch 22/50 train=0.6213 val=0.6183
2026-05-01 09:28:02,523 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:28:02,523 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:28:02,523 INFO train_multi TF=ALL: new best val=0.6183 — saved
2026-05-01 09:28:18,474 INFO train_multi TF=ALL epoch 23/50 train=0.6177 val=0.6164
2026-05-01 09:28:18,481 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:28:18,481 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:28:18,481 INFO train_multi TF=ALL: new best val=0.6164 — saved
2026-05-01 09:28:34,327 INFO train_multi TF=ALL epoch 24/50 train=0.6156 val=0.6130
2026-05-01 09:28:34,333 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:28:34,333 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:28:34,333 INFO train_multi TF=ALL: new best val=0.6130 — saved
2026-05-01 09:28:50,211 INFO train_multi TF=ALL epoch 25/50 train=0.6132 val=0.6203
2026-05-01 09:29:05,731 INFO train_multi TF=ALL epoch 26/50 train=0.6112 val=0.6133
2026-05-01 09:29:21,338 INFO train_multi TF=ALL epoch 27/50 train=0.6089 val=0.6156
2026-05-01 09:29:36,689 INFO train_multi TF=ALL epoch 28/50 train=0.6074 val=0.6140
2026-05-01 09:29:52,241 INFO train_multi TF=ALL epoch 29/50 train=0.6055 val=0.6136
2026-05-01 09:30:07,877 INFO train_multi TF=ALL epoch 30/50 train=0.6035 val=0.6098
2026-05-01 09:30:07,883 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:30:07,883 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:30:07,883 INFO train_multi TF=ALL: new best val=0.6098 — saved
2026-05-01 09:30:23,483 INFO train_multi TF=ALL epoch 31/50 train=0.6024 val=0.6113
2026-05-01 09:30:39,353 INFO train_multi TF=ALL epoch 32/50 train=0.6008 val=0.6125
2026-05-01 09:30:55,085 INFO train_multi TF=ALL epoch 33/50 train=0.5997 val=0.6112
2026-05-01 09:31:10,882 INFO train_multi TF=ALL epoch 34/50 train=0.5986 val=0.6093
2026-05-01 09:31:10,888 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:31:10,888 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:31:10,888 INFO train_multi TF=ALL: new best val=0.6093 — saved
2026-05-01 09:31:26,677 INFO train_multi TF=ALL epoch 35/50 train=0.5962 val=0.6088
2026-05-01 09:31:26,683 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:31:26,683 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:31:26,683 INFO train_multi TF=ALL: new best val=0.6088 — saved
2026-05-01 09:31:42,537 INFO train_multi TF=ALL epoch 36/50 train=0.5952 val=0.6154
2026-05-01 09:31:58,401 INFO train_multi TF=ALL epoch 37/50 train=0.5941 val=0.6104
2026-05-01 09:32:14,131 INFO train_multi TF=ALL epoch 38/50 train=0.5923 val=0.6105
2026-05-01 09:32:30,091 INFO train_multi TF=ALL epoch 39/50 train=0.5921 val=0.6104
2026-05-01 09:32:45,876 INFO train_multi TF=ALL epoch 40/50 train=0.5901 val=0.6108
2026-05-01 09:33:01,439 INFO train_multi TF=ALL epoch 41/50 train=0.5888 val=0.6105
2026-05-01 09:33:17,360 INFO train_multi TF=ALL epoch 42/50 train=0.5877 val=0.6157
2026-05-01 09:33:32,835 INFO train_multi TF=ALL epoch 43/50 train=0.5864 val=0.6074
2026-05-01 09:33:32,841 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:33:32,841 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:33:32,841 INFO train_multi TF=ALL: new best val=0.6074 — saved
2026-05-01 09:33:48,439 INFO train_multi TF=ALL epoch 44/50 train=0.5857 val=0.6097
2026-05-01 09:34:04,017 INFO train_multi TF=ALL epoch 45/50 train=0.5840 val=0.6099
2026-05-01 09:34:20,059 INFO train_multi TF=ALL epoch 46/50 train=0.5829 val=0.6056
2026-05-01 09:34:20,065 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:34:20,065 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:34:20,066 INFO train_multi TF=ALL: new best val=0.6056 — saved
2026-05-01 09:34:35,704 INFO train_multi TF=ALL epoch 47/50 train=0.5815 val=0.6118
2026-05-01 09:34:51,540 INFO train_multi TF=ALL epoch 48/50 train=0.5803 val=0.6121
2026-05-01 09:35:07,261 INFO train_multi TF=ALL epoch 49/50 train=0.5796 val=0.6197
2026-05-01 09:35:22,792 INFO train_multi TF=ALL epoch 50/50 train=0.5785 val=0.6149
2026-05-01 09:35:22,972 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-05-01 09:35:22,973 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-05-01 09:35:22,973 INFO Retrain complete. Total wall-clock: 880.7s
2026-05-01 09:35:26,202 INFO Model gru: SUCCESS
2026-05-01 09:35:26,203 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:35:26,203 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-01 09:35:26,203 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-01 09:35:26,203 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-01 09:35:26,203 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-05-01 09:35:26,203 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-01 09:35:26,204 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-05-01 09:35:26,205 INFO Saved 8 retrain records to metrics/

=== TRAINING COMPLETE ===
  regime: SUCCESS
  gru: SUCCESS
  DONE  Step 7a - GRU+Regime

=== Clean Quality/RL source: Backtest on train window ===
  START Train-window backtest for Quality/RL labels
2026-05-01 09:35:27,009 INFO === STEP 6: BACKTEST (train) ===
2026-05-01 09:35:27,010 INFO BT_WINDOW=train — train-window backtest: 2016-01-04 → 2021-08-05 (clean Quality/RL labels)
2026-05-01 09:35:27,010 INFO ================================================================
  ROUND 0 / 3
================================================================
2026-05-01 09:35:27,010 INFO Round 0 — running backtest: 2016-01-04 → 2021-08-05 (ml_trader, shared ML cache)
2026-05-01 09:35:29,759 WARNING QualityScorer unavailable (weights missing or load failed)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
2026-05-01 09:43:05,857 WARNING ml_trader: portfolio drawdown 100.6% after trade exit — halting all trading

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260501_093529.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*  Sortino     IC  Verdict
-----------------------------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)              992  25.0%   0.86 -100.7%  -0.102 25.0%  7.9% 100.6%    -1.00    -0.10 -0.068     FAIL
  FAILED rules: positive_expectancy, profit_factor_min_1_25, drawdown_below_20pct, sharpe_positive, sortino_positive, win_rate_above_breakeven, mc_p10_not_ruin, sharpe_ci_positive
  monthly R: 2016-10=-7.64  2016-11=-48.85  2016-12=-18.27  2017-01=+6.80  2017-02=-16.73  2017-03=-29.61
  MonteCarlo P95 DD=128.5%  P10 equity=-70  t=-1.98 (p=0.048)  Sharpe CI=[-2.08, -0.12]  streak=23
  gate_diagnostics: bars=317300 no_signal=148734 quality_block=0 session_skip=167000 density=574 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: high_uncertainty=89887, htf_bias_conflict=30687, neutral_requires_ltf_ranging=14450, volatile_structure_missing=9093, volatile_weak_conf=2433, trend_structure_missing=1445

Calibration Summary:
  all          [OK] Too few populated bins for calibration check
  ml_trader    [OK] Too few populated bins for calibration check
2026-05-01 09:43:07,872 INFO Round 0 backtest — 992 trades | avg WR=25.0% | avg PF=0.86 | avg Sharpe=-1.00
2026-05-01 09:43:07,872 INFO   ml_trader: 992 trades | WR=25.0% | fixed PF=0.86 | Return=-100.7% | ExpR=-0.102 | DD=100.6% | Sharpe=-1.00
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_0.json
INFO  Total trades: 992
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (992 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD         87 trades    43 days   2.02/day  [OVERTRADE]
  EURGBP        131 trades    60 days   2.18/day  [OVERTRADE]
  EURJPY         87 trades    44 days   1.98/day  [OVERTRADE]
  EURUSD        104 trades    51 days   2.04/day  [OVERTRADE]
  GBPJPY         73 trades    35 days   2.09/day  [OVERTRADE]
  GBPUSD         79 trades    36 days   2.19/day  [OVERTRADE]
  NZDUSD        104 trades    51 days   2.04/day  [OVERTRADE]
  USDCAD        101 trades    53 days   1.91/day  [OVERTRADE]
  USDCHF        107 trades    61 days   1.75/day  [OVERTRADE]
  USDJPY         63 trades    34 days   1.85/day  [OVERTRADE]
  XAUUSD         56 trades    29 days   1.93/day  [OVERTRADE]
  ⚠  AUDUSD: 2.02/day (>1.5)
  ⚠  EURGBP: 2.18/day (>1.5)
  ⚠  EURJPY: 1.98/day (>1.5)
  ⚠  EURUSD: 2.04/day (>1.5)
  ⚠  GBPJPY: 2.09/day (>1.5)
  ⚠  GBPUSD: 2.19/day (>1.5)
  ⚠  NZDUSD: 2.04/day (>1.5)
  ⚠  USDCAD: 1.91/day (>1.5)
  ⚠  USDCHF: 1.75/day (>1.5)
  ⚠  USDJPY: 1.85/day (>1.5)
  ⚠  XAUUSD: 1.93/day (>1.5)

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN           733 trades   73.9%  WR=25.2%  avgEV=0.000
  BIAS_UP             259 trades   26.1%  WR=24.3%  avgEV=0.000
  ⚠  BIAS_DOWN = 74% of trades — regime collapse?
  ⚠  Regimes never traded: ['BIAS_NEUTRAL', 'CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']

──────────────────────────────────────────────────────────────
CHECK 3: EV PREDICTED vs REALIZED RR
──────────────────────────────────────────────────────────────
  Pearson  = +nan   Spearman = -0.0771

  Bucket                  N     AvgEV     AvgRR   WinRate
  Q1 (low EV)             0       n/a       n/a       n/a
  Q2                      0       n/a       n/a       n/a
  Q3                      0       n/a       n/a       n/a
  Q4 (high EV)          992     0.000    -0.102     25.0%

  Top-20% EV trades: n=992  avgEV=0.0  avgRR=-0.102  WR=25.0%

  Per-regime EV↔RR correlation:
  Regime                N    Pearson   Spearman       WR     AvgEV
  BIAS_DOWN           733       +nan    -0.0815   25.2%     0.000
  BIAS_UP             259       +nan    +0.1071   24.3%     0.000
  ⚠  EV↔RR Spearman=-0.077 < 0.15 — EV rankings don't predict outcomes
  ⚠  Top-20% EV trades win_rate=25.0% — high-EV selection not working
  ⚠  EV↔RR Spearman in BIAS_DOWN = -0.082 — EV useless in this regime

──────────────────────────────────────────────────────────────
CHECK 4: GRU CONFIDENCE CALIBRATION  (p_bull/bear vs win rate)
──────────────────────────────────────────────────────────────
  ECE = 0.5615  (target < 0.10)
  Bin                   N   Midpoint    WinRate    Error
  [0.68-0.74]         151      0.709      0.258    0.451
  [0.74-0.79]         211      0.765      0.265    0.500
  [0.79-0.85]         326      0.822      0.258    0.564
  [0.85-0.91]         274      0.878      0.212    0.666
  [0.91-0.96]          30      0.934      0.367    0.567
  ⚠  Bin [0.68-0.74]: midpoint=0.71 win_rate=0.26 (err=0.45 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.74-0.79]: midpoint=0.77 win_rate=0.27 (err=0.50 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.79-0.85]: midpoint=0.82 win_rate=0.26 (err=0.56 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.85-0.91]: midpoint=0.88 win_rate=0.21 (err=0.67 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.91-0.96]: midpoint=0.93 win_rate=0.37 (err=0.57 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable

──────────────────────────────────────────────────────────────
CHECK 5: GRU ↔ EV MODEL CONSISTENCY  (direction × value agreement)
──────────────────────────────────────────────────────────────
  Pearson=+nan  Spearman=-0.0402  Agree=50%

  Quadrants  (conf_threshold=median, ev_threshold=median):
  high_conf + high_ev:   496  ← ideal
  high_conf + low_ev:      0  ← GRU overconfident
  low_conf  + high_ev:   496  ← EV optimistic
  low_conf  + low_ev:      0  ← correct abstention
  ⚠  GRU and EV agree on only 50.0% of trades — models pulling in opposite directions

──────────────────────────────────────────────────────────────
SUMMARY — 23 flag(s):
  ⚠  AUDUSD: 2.02/day (>1.5)
  ⚠  EURGBP: 2.18/day (>1.5)
  ⚠  EURJPY: 1.98/day (>1.5)
  ⚠  EURUSD: 2.04/day (>1.5)
  ⚠  GBPJPY: 2.09/day (>1.5)
  ⚠  GBPUSD: 2.19/day (>1.5)
  ⚠  NZDUSD: 2.04/day (>1.5)
  ⚠  USDCAD: 1.91/day (>1.5)
  ⚠  USDCHF: 1.75/day (>1.5)
  ⚠  USDJPY: 1.85/day (>1.5)
  ⚠  XAUUSD: 1.93/day (>1.5)
  ⚠  BIAS_DOWN = 74% of trades — regime collapse?
  ⚠  Regimes never traded: ['BIAS_NEUTRAL', 'CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']
  ⚠  EV↔RR Spearman=-0.077 < 0.15 — EV rankings don't predict outcomes
  ⚠  Top-20% EV trades win_rate=25.0% — high-EV selection not working
  ⚠  EV↔RR Spearman in BIAS_DOWN = -0.082 — EV useless in this regime
  ⚠  Bin [0.68-0.74]: midpoint=0.71 win_rate=0.26 (err=0.45 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.74-0.79]: midpoint=0.77 win_rate=0.27 (err=0.50 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.79-0.85]: midpoint=0.82 win_rate=0.26 (err=0.56 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.85-0.91]: midpoint=0.88 win_rate=0.21 (err=0.67 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.91-0.96]: midpoint=0.93 win_rate=0.37 (err=0.57 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable
  ⚠  GRU and EV agree on only 50.0% of trades — models pulling in opposite directions
──────────────────────────────────────────────────────────────
2026-05-01 09:43:08,573 INFO Round 0: wrote 992 journal entries (total in file: 992)

======================================================================
  BACKTEST COMPLETE  (round 0 / window=train)
======================================================================
  Round     Trades       WR     PF*  Sharpe*
  ------------------------------------------
  Round 0        992     25.0%    0.865    -0.999

  DONE  Train-window backtest for Quality/RL labels
  Saved Train Quality/RL source result → train_quality_rl_source_summary.json
  Train-label journal entries: 992

=== Train Quality + RL on train-only journal ===
  START Train-only Quality+RL retrain
2026-05-01 09:43:08,831 INFO === STEP 7b: QUALITY + RL TRAINING ===
2026-05-01 09:43:08,857 INFO Journal entries: 992 total, 992 allowed for training (['live', 'paper', 'production', 'train'])
2026-05-01 09:43:08,857 INFO --- Training quality ---
2026-05-01 09:43:08,857 INFO Running retrain --model quality with JOURNAL_ALLOWED_SPLITS=train,live,paper,production
2026-05-01 09:43:09,073 INFO retrain environment: KAGGLE
2026-05-01 09:43:10,936 INFO Device: CUDA (2 GPU(s))
2026-05-01 09:43:10,949 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 09:43:10,949 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 09:43:10,949 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-01 09:43:10,949 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-01 09:43:10,950 INFO Retrain data split: train
2026-05-01 09:43:10,951 INFO === QualityScorer retrain ===
2026-05-01 09:43:11,120 INFO NumExpr defaulting to 4 threads.
2026-05-01 09:43:11,358 INFO QualityScorer: CUDA available — using GPU
2026-05-01 09:43:11,431 INFO Quality phase label creation: 0.1s (992 trades)
2026-05-01 09:43:11,506 INFO QualityScorer: 992 samples, EV stats={'mean': -0.36950603127479553, 'std': 1.1503156423568726, 'n_pos': 248, 'n_neg': 744}, device=cuda
2026-05-01 09:43:11,507 INFO QualityScorer: normalised win labels by median_win=1.130 — EV range now [-1, +3]
2026-05-01 09:43:11,730 INFO QualityScorer: DataParallel across 2 GPUs
2026-05-01 09:43:11,730 INFO QualityScorer: cold start
2026-05-01 09:43:11,731 INFO QualityScorer: pos_weight=2.87 (n_pos=205 n_neg=588)
2026-05-01 09:43:14,278 INFO Quality epoch   1/100 — va_huber=0.9720
2026-05-01 09:43:14,335 INFO Quality epoch   2/100 — va_huber=0.9671
2026-05-01 09:43:14,374 INFO Quality epoch   3/100 — va_huber=0.9631
2026-05-01 09:43:14,622 INFO Quality epoch   4/100 — va_huber=0.9599
2026-05-01 09:43:14,658 INFO Quality epoch   5/100 — va_huber=0.9569
2026-05-01 09:43:15,292 INFO Quality epoch  11/100 — va_huber=0.9493
2026-05-01 09:43:15,666 INFO Quality epoch  21/100 — va_huber=0.9242
2026-05-01 09:43:16,048 INFO Quality epoch  31/100 — va_huber=0.9181
2026-05-01 09:43:16,408 INFO Quality early stop at epoch 40
2026-05-01 09:43:16,421 INFO QualityScorer EV model: MAE=0.989 dir_acc=0.588 n_val=199
2026-05-01 09:43:16,428 INFO QualityScorer saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-05-01 09:43:16,495 INFO Quality phase train: 5.1s | total: 5.5s
2026-05-01 09:43:16,507 INFO Retrain complete. Total wall-clock: 5.6s
2026-05-01 09:43:17,812 INFO Model quality: SUCCESS
2026-05-01 09:43:17,812 INFO --- Training rl ---
2026-05-01 09:43:17,812 INFO Running retrain --model rl with JOURNAL_ALLOWED_SPLITS=train,live,paper,production
2026-05-01 09:43:18,027 INFO retrain environment: KAGGLE
2026-05-01 09:43:19,913 INFO Device: CUDA (2 GPU(s))
2026-05-01 09:43:19,925 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 09:43:19,926 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 09:43:19,926 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-01 09:43:19,926 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-01 09:43:19,926 INFO Retrain data split: train
2026-05-01 09:43:19,928 INFO === RLAgent (PPO) retrain ===
2026-05-01 09:43:19,935 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/rl_ppo_20260501_094319
2026-05-01 09:43:19,975 INFO RL phase episode loading: 0.0s (992 episodes)
2026-05-01 09:43:23.985525: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1777628604.242643   57806 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1777628604.309924   57806 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1777628604.908324   57806 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777628604.908396   57806 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777628604.908400   57806 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1777628604.908403   57806 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-05-01 09:43:41,218 INFO NumExpr defaulting to 4 threads.
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
2026-05-01 09:43:45,342 INFO RLAgent: cold start — building new PPO policy
2026-05-01 09:44:08,597 INFO RLAgent: retrain complete, 992 episodes
2026-05-01 09:44:08,598 INFO RL phase PPO train: 48.6s | total: 48.7s
2026-05-01 09:44:08,613 INFO Retrain complete. Total wall-clock: 48.7s
2026-05-01 09:44:10,657 INFO Model rl: SUCCESS
2026-05-01 09:44:10,658 INFO Step 7b complete — summary: /kaggle/working/Multi-Bot/trading-system/ml_training/metrics/training_7b_summary.json
  DONE  Train-only Quality+RL retrain
  Archived journal → trade_journal_train_only.jsonl
  Archived journal CSV → trade_journal_train_only.csv

=== Round 1: Backtest on validation window (last 2yr of training data) ===
  Cleared journal for fresh Round 1 run
  START Round 1 - Backtest (val)
2026-05-01 09:44:11,305 INFO === STEP 6: BACKTEST (round1) ===
2026-05-01 09:44:11,306 INFO BT_WINDOW=round1 — val-window backtest: 2021-08-05 → 2023-08-04 (test set protected)
2026-05-01 09:44:11,306 INFO ================================================================
  ROUND 1 / 3
================================================================
2026-05-01 09:44:11,307 INFO Round 1 — running backtest: 2021-08-05 → 2023-08-04 (ml_trader, shared ML cache)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
2026-05-01 09:47:23,309 WARNING ml_trader: portfolio drawdown 100.1% after trade exit — halting all trading

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260501_094413.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*  Sortino     IC  Verdict
-----------------------------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)              336  20.5%   0.63 -100.1%  -0.298 20.5%  4.8% 100.1%    -3.29    -0.30 -0.018     FAIL
  FAILED rules: positive_expectancy, profit_factor_min_1_25, drawdown_below_20pct, sharpe_positive, sortino_positive, win_rate_above_breakeven, mc_p10_not_ruin, sharpe_ci_positive
  monthly R: 2022-01=-14.96  2022-02=-26.46  2022-03=-11.37  2022-04=-6.50  2022-05=-16.18  2022-06=-23.82
  MonteCarlo P95 DD=109.5%  P10 equity=-11  t=-3.80 (p=0.000)  Sharpe CI=[-5.37, -1.43]  streak=28
  gate_diagnostics: bars=226017 no_signal=105174 quality_block=1306 session_skip=118999 density=202 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: high_uncertainty=62597, htf_bias_conflict=21852, volatile_structure_missing=12168, neutral_requires_ltf_ranging=3812, volatile_weak_conf=3276, trend_structure_missing=858

Calibration Summary:
  all          [WARN] Non-monotonic calibration: 3/6 pairs violated. Consider retraining QualityScorer
  ml_trader    [WARN] Non-monotonic calibration: 3/6 pairs violated. Consider retraining QualityScorer
2026-05-01 09:47:25,303 INFO Round 1 backtest — 336 trades | avg WR=20.5% | avg PF=0.62 | avg Sharpe=-3.29
2026-05-01 09:47:25,304 INFO   ml_trader: 336 trades | WR=20.5% | fixed PF=0.62 | Return=-100.1% | ExpR=-0.298 | DD=100.1% | Sharpe=-3.29
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_1.json
INFO  Total trades: 336
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (336 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD         28 trades    16 days   1.75/day  [OVERTRADE]
  EURGBP         21 trades    13 days   1.61/day  [OVERTRADE]
  EURJPY         24 trades    15 days   1.60/day  [OVERTRADE]
  EURUSD         15 trades     9 days   1.67/day  [OVERTRADE]
  GBPJPY         59 trades    26 days   2.27/day  [OVERTRADE]
  GBPUSD         23 trades    11 days   2.09/day  [OVERTRADE]
  NZDUSD         35 trades    18 days   1.94/day  [OVERTRADE]
  USDCAD         50 trades    25 days   2.00/day  [OVERTRADE]
  USDCHF         27 trades    17 days   1.59/day  [OVERTRADE]
  USDJPY         18 trades    10 days   1.80/day  [OVERTRADE]
  XAUUSD         36 trades    21 days   1.71/day  [OVERTRADE]
  ⚠  AUDUSD: 1.75/day (>1.5)
  ⚠  EURGBP: 1.62/day (>1.5)
  ⚠  EURJPY: 1.60/day (>1.5)
  ⚠  EURUSD: 1.67/day (>1.5)
  ⚠  GBPJPY: 2.27/day (>1.5)
  ⚠  GBPUSD: 2.09/day (>1.5)
  ⚠  NZDUSD: 1.94/day (>1.5)
  ⚠  USDCAD: 2.00/day (>1.5)
  ⚠  USDCHF: 1.59/day (>1.5)
  ⚠  USDJPY: 1.80/day (>1.5)
  ⚠  XAUUSD: 1.71/day (>1.5)

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN           294 trades   87.5%  WR=22.1%  avgEV=0.532
  BIAS_UP              42 trades   12.5%  WR=9.5%  avgEV=0.636
  ⚠  BIAS_DOWN = 88% of trades — regime collapse?
  ⚠  Regimes never traded: ['BIAS_NEUTRAL', 'CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']

──────────────────────────────────────────────────────────────
CHECK 3: EV PREDICTED vs REALIZED RR
──────────────────────────────────────────────────────────────
  Pearson  = -0.1013   Spearman = -0.0580

  Bucket                  N     AvgEV     AvgRR   WinRate
  Q1 (low EV)            84     0.123    -0.364     19.0%
  Q2                     84     0.368    -0.077     26.2%
  Q3                     84     0.595    -0.059     28.6%
  Q4 (high EV)           84     1.095    -0.692      8.3%

  Top-20% EV trades: n=68  avgEV=1.172  avgRR=-0.775  WR=5.9%

  Per-regime EV↔RR correlation:
  Regime                N    Pearson   Spearman       WR     AvgEV
  BIAS_DOWN           294    -0.1096    -0.0721   22.1%     0.532
  BIAS_UP              42    -0.0066    -0.1122    9.5%     0.636
  ⚠  EV↔RR Pearson=-0.101 < 0.1 — EV model weak, check training labels
  ⚠  EV↔RR Spearman=-0.058 < 0.15 — EV rankings don't predict outcomes
  ⚠  Non-monotonic bins: Q4 avg_rr=-0.692 ≤ Q1 avg_rr=-0.364 — EV not predictive
  ⚠  Top-20% EV trades win_rate=5.9% — high-EV selection not working
  ⚠  EV↔RR Spearman in BIAS_DOWN = -0.072 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_UP = -0.112 — EV useless in this regime

──────────────────────────────────────────────────────────────
CHECK 4: GRU CONFIDENCE CALIBRATION  (p_bull/bear vs win rate)
──────────────────────────────────────────────────────────────
  ECE = 0.5924  (target < 0.10)
  Bin                   N   Midpoint    WinRate    Error
  [0.66-0.72]          28      0.689      0.286    0.403
  [0.72-0.77]          87      0.742      0.230    0.512
  [0.77-0.82]          92      0.795      0.163    0.632
  [0.82-0.88]          99      0.849      0.182    0.667
  [0.88-0.93]          30      0.902      0.267    0.635
  ⚠  Bin [0.66-0.72]: midpoint=0.69 win_rate=0.29 (err=0.40 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.72-0.77]: midpoint=0.74 win_rate=0.23 (err=0.51 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.77-0.82]: midpoint=0.80 win_rate=0.16 (err=0.63 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.82-0.88]: midpoint=0.85 win_rate=0.18 (err=0.67 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.88-0.93]: midpoint=0.90 win_rate=0.27 (err=0.64 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable

──────────────────────────────────────────────────────────────
CHECK 5: GRU ↔ EV MODEL CONSISTENCY  (direction × value agreement)
──────────────────────────────────────────────────────────────
  Pearson=+0.0154  Spearman=+0.0180  Agree=55%

  Quadrants  (conf_threshold=median, ev_threshold=median):
  high_conf + high_ev:    93  ← ideal
  high_conf + low_ev:     75  ← GRU overconfident
  low_conf  + high_ev:    75  ← EV optimistic
  low_conf  + low_ev:     93  ← correct abstention
  ⚠  GRU↔EV Pearson=0.015 < 0.1 — direction model and EV model disagree (architecture misaligned?)

──────────────────────────────────────────────────────────────
SUMMARY — 26 flag(s):
  ⚠  AUDUSD: 1.75/day (>1.5)
  ⚠  EURGBP: 1.62/day (>1.5)
  ⚠  EURJPY: 1.60/day (>1.5)
  ⚠  EURUSD: 1.67/day (>1.5)
  ⚠  GBPJPY: 2.27/day (>1.5)
  ⚠  GBPUSD: 2.09/day (>1.5)
  ⚠  NZDUSD: 1.94/day (>1.5)
  ⚠  USDCAD: 2.00/day (>1.5)
  ⚠  USDCHF: 1.59/day (>1.5)
  ⚠  USDJPY: 1.80/day (>1.5)
  ⚠  XAUUSD: 1.71/day (>1.5)
  ⚠  BIAS_DOWN = 88% of trades — regime collapse?
  ⚠  Regimes never traded: ['BIAS_NEUTRAL', 'CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']
  ⚠  EV↔RR Pearson=-0.101 < 0.1 — EV model weak, check training labels
  ⚠  EV↔RR Spearman=-0.058 < 0.15 — EV rankings don't predict outcomes
  ⚠  Non-monotonic bins: Q4 avg_rr=-0.692 ≤ Q1 avg_rr=-0.364 — EV not predictive
  ⚠  Top-20% EV trades win_rate=5.9% — high-EV selection not working
  ⚠  EV↔RR Spearman in BIAS_DOWN = -0.072 — EV useless in this regime
  ⚠  EV↔RR Spearman in BIAS_UP = -0.112 — EV useless in this regime
  ⚠  Bin [0.66-0.72]: midpoint=0.69 win_rate=0.29 (err=0.40 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.72-0.77]: midpoint=0.74 win_rate=0.23 (err=0.51 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.77-0.82]: midpoint=0.80 win_rate=0.16 (err=0.63 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.82-0.88]: midpoint=0.85 win_rate=0.18 (err=0.67 > 0.15) — GRU miscalibrated
  ⚠  Bin [0.88-0.93]: midpoint=0.90 win_rate=0.27 (err=0.64 > 0.15) — GRU miscalibrated
  ⚠  Win rate non-monotonic across confidence bins — GRU confidence unreliable
  ⚠  GRU↔EV Pearson=0.015 < 0.1 — direction model and EV model disagree (architecture misaligned?)
──────────────────────────────────────────────────────────────
2026-05-01 09:47:25,761 INFO Round 1: wrote 336 journal entries (total in file: 336)

======================================================================
  BACKTEST COMPLETE  (round 1 / window=round1)
======================================================================
  Round     Trades       WR     PF*  Sharpe*
  ------------------------------------------
  Round 1        336     20.5%    0.625    -3.289

  DONE  Round 1 - Backtest (val)
  Saved Round 1 result → round1_summary.json
  Journal after Round 1: 336 entries

  SKIP  Round 1 Quality+RL retrain — validation journal kept evaluation-only

=== Round 2: BLIND backtest on test window (unseen 2yr) ===
  START Round 2 - Blind backtest (test)
2026-05-01 09:47:26,583 INFO === STEP 6: BACKTEST (round2) ===
2026-05-01 09:47:26,585 INFO BT_WINDOW=round2 — BLIND backtest: 2023-08-07 → 2025-08-05 (test set)
2026-05-01 09:47:26,585 INFO ================================================================
  ROUND 2 / 3
================================================================
2026-05-01 09:47:26,585 INFO Round 2 — running backtest: 2023-08-07 → 2025-08-05 (ml_trader, shared ML cache)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()

Backtest results → /kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../backtest_results/backtest_20260501_094729.json
Trader                                   Trades      WR    PF*  Return*    ExpR   TP1%   TP2%     DD*  Sharpe*  Sortino     IC  Verdict
-----------------------------------------------------------------------------------------------------------------------------
ML-Native Execution (GRU + EV)              652  25.3%   0.87  -62.4%  -0.096 25.3%  6.9%  77.5%    -0.95    -0.10 -0.059     FAIL
  FAILED rules: positive_expectancy, profit_factor_min_1_25, drawdown_below_20pct, sharpe_positive, sortino_positive, win_rate_above_breakeven, mc_p10_not_ruin, sharpe_ci_positive
  monthly R: 2025-02=+21.39  2025-03=-12.80  2025-04=-16.72  2025-05=-9.31  2025-06=-17.40  2025-07=-23.05
  MonteCarlo P95 DD=88.6%  P10 equity=3,759  t=-1.53 (p=0.127)  Sharpe CI=[-2.29, 0.22]  streak=27
  gate_diagnostics: bars=482221 no_signal=223435 quality_block=2948 session_skip=254850 density=336 pm_reject=0 daily_skip=0 cooldown=0 daily_halt_events=0 enforce_daily_halt=False
  no_signal_reasons: high_uncertainty=132660, htf_bias_conflict=45092, volatile_structure_missing=26091, neutral_requires_ltf_ranging=9672, volatile_weak_conf=6877, trend_structure_missing=1932

Calibration Summary:
  all          [WARN] Non-monotonic calibration: 2/6 pairs violated. Consider retraining QualityScorer
  ml_trader    [WARN] Non-monotonic calibration: 2/6 pairs violated. Consider retraining QualityScorer
2026-05-01 09:51:22,376 INFO Round 2 backtest — 652 trades | avg WR=25.3% | avg PF=0.87 | avg Sharpe=-0.95
2026-05-01 09:51:22,376 INFO   ml_trader: 652 trades | WR=25.3% | fixed PF=0.87 | Return=-62.4% | ExpR=-0.096 | DD=77.5% | Sharpe=-0.95
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_2.json
INFO  Total trades: 652
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (652 rows)

──────────────────────────────────────────────────────────────
CHECK 1: TRADE FREQUENCY  (trades/day/symbol)
──────────────────────────────────────────────────────────────
  AUDUSD         91 trades    48 days   1.90/day  [OVERTRADE]
  EURGBP          7 trades     4 days   1.75/day  [OVERTRADE]
  EURJPY         77 trades    40 days   1.93/day  [OVERTRADE]
  EURUSD         35 trades    21 days   1.67/day  [OVERTRADE]
  GBPJPY         98 trades    52 days   1.89/day  [OVERTRADE]
  GBPUSD         53 trades    38 days   1.40/day
  NZDUSD         38 trades    18 days   2.11/day  [OVERTRADE]
  USDCAD         47 trades    22 days   2.14/day  [OVERTRADE]
  USDCHF         38 trades    19 days   2.00/day  [OVERTRADE]
  USDJPY         40 trades    22 days   1.82/day  [OVERTRADE]
  XAUUSD        128 trades    69 days   1.85/day  [OVERTRADE]
  ⚠  AUDUSD: 1.90/day (>1.5)
  ⚠  EURGBP: 1.75/day (>1.5)
  ⚠  EURJPY: 1.92/day (>1.5)
  ⚠  EURUSD: 1.67/day (>1.5)
  ⚠  GBPJPY: 1.88/day (>1.5)
  ⚠  NZDUSD: 2.11/day (>1.5)
  ⚠  USDCAD: 2.14/day (>1.5)
  ⚠  USDCHF: 2.00/day (>1.5)
  ⚠  USDJPY: 1.82/day (>1.5)
  ⚠  XAUUSD: 1.86/day (>1.5)

──────────────────────────────────────────────────────────────
CHECK 2: REGIME DISTRIBUTION  (% of trades, WR, avgEV)
──────────────────────────────────────────────────────────────
  BIAS_DOWN           585 trades   89.7%  WR=24.8%  avgEV=0.561
  BIAS_UP              67 trades   10.3%  WR=29.9%  avgEV=0.619
  ⚠  BIAS_DOWN = 90% of trades — regime collapse?
  ⚠  Regimes never traded: ['BIAS_NEUTRAL', 'CONSOLIDATING', 'RANGING', 'TRENDING', 'VOLATILE']

──────────────────────────────────────────────────────────────
CHECK 3: EV PREDICTED vs REALIZED RR
──────────────────────────────────────────────────────────────
  Pearson  = -0.0555   Spearman = -0.0265

  Bucket                  N     AvgEV     AvgRR   WinRate
  Q1 (low EV)           163     0.141    -0.047     24.5%
  Q2                    163     0.390    -0.029     26.4%
  Q3                    163     0.624     0.068     31.3%
  Q4 (high EV)          163     1.111    -0.376     19.0%

  Top-20% EV trades: n=131  avgEV=1.188  avgRR=-0.328  WR=20.6%

  Per-regime EV↔RR correlation:
  Regime                N    Pearson   Spearm
2026-05-01 09:51:22,961 INFO Round 2: wrote 652 journal entries (total in file: 988)
  DONE  Round 2 - Blind backtest (test)
  Saved Round 2 result → round2_summary.json
  Journal after Round 2: 988 entries

  SKIP  Round 2 Quality+RL retrain — blind test journal kept untouched

=== Round 3: Incremental retrain ===
  START Retrain gru [train-split retrain]
2026-05-01 09:51:23,341 INFO retrain environment: KAGGLE
2026-05-01 09:51:25,255 INFO Device: CUDA (2 GPU(s))
2026-05-01 09:51:25,267 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 09:51:25,267 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 09:51:25,267 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-01 09:51:25,268 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-01 09:51:25,268 INFO Retrain data split: train
2026-05-01 09:51:25,269 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-05-01 09:51:25,439 INFO NumExpr defaulting to 4 threads.
2026-05-01 09:51:25,683 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-01 09:51:25,683 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 09:51:25,683 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 09:51:25,942 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-05-01 09:51:25,942 INFO GRU phase macro_correlations: 0.0s
2026-05-01 09:51:25,942 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-05-01 09:51:25,944 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260501_095125
2026-05-01 09:51:25,948 INFO GRU feature contract unchanged (input_size=78) — incremental retrain
2026-05-01 09:51:26,118 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:26,143 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:26,161 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:26,169 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:26,171 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-01 09:51:26,171 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 09:51:26,171 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 09:51:26,172 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-05-01 09:51:26,173 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:26,308 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 17, 'BIAS_DOWN': 25, 'BIAS_NEUTRAL': 8360}  ambiguous=8312 (total=8402)  short_runs_zeroed=15
2026-05-01 09:51:26,310 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:26,640 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 619, 'RANGING': 24879, 'CONSOLIDATING': 1882, 'VOLATILE': 5358}  ambiguous=26508 (total=32738)  short_runs_zeroed=2571
2026-05-01 09:51:26,671 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:27,013 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:27,178 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:27,316 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:27,566 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:27,588 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:27,605 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:27,614 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:27,615 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:27,714 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 100, 'BIAS_DOWN': 31, 'BIAS_NEUTRAL': 8271}  ambiguous=8272 (total=8402)  short_runs_zeroed=52
2026-05-01 09:51:27,716 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:27,996 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 752, 'RANGING': 24932, 'CONSOLIDATING': 2099, 'VOLATILE': 4955}  ambiguous=26224 (total=32738)  short_runs_zeroed=2112
2026-05-01 09:51:28,016 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:28,341 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:28,506 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:28,639 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:28,876 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:28,898 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:28,914 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:28,923 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:28,924 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:29,024 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 46, 'BIAS_DOWN': 4, 'BIAS_NEUTRAL': 8352}  ambiguous=8316 (total=8402)  short_runs_zeroed=28
2026-05-01 09:51:29,026 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:29,314 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 617, 'RANGING': 24738, 'CONSOLIDATING': 2103, 'VOLATILE': 5282}  ambiguous=26101 (total=32740)  short_runs_zeroed=2218
2026-05-01 09:51:29,331 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:29,666 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:29,827 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:29,956 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:30,193 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:30,214 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:30,232 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:30,240 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:30,241 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:30,346 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 15, 'BIAS_DOWN': 19, 'BIAS_NEUTRAL': 8368}  ambiguous=8331 (total=8402)  short_runs_zeroed=27
2026-05-01 09:51:30,348 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:30,633 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 842, 'RANGING': 25174, 'CONSOLIDATING': 1996, 'VOLATILE': 4727}  ambiguous=26490 (total=32739)  short_runs_zeroed=2045
2026-05-01 09:51:30,660 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:31,002 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:31,165 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:31,299 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:31,535 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:31,558 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:31,576 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:31,586 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:31,587 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:31,687 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 93, 'BIAS_DOWN': 58, 'BIAS_NEUTRAL': 8252}  ambiguous=8287 (total=8403)  short_runs_zeroed=75
2026-05-01 09:51:31,690 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:31,968 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 761, 'RANGING': 24579, 'CONSOLIDATING': 2089, 'VOLATILE': 5311}  ambiguous=25954 (total=32740)  short_runs_zeroed=2172
2026-05-01 09:51:31,986 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:32,320 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:32,495 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:32,632 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:32,872 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:32,894 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:32,911 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:32,919 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:32,921 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:33,020 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 87, 'BIAS_DOWN': 55, 'BIAS_NEUTRAL': 8261}  ambiguous=8278 (total=8403)  short_runs_zeroed=63
2026-05-01 09:51:33,023 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:33,313 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 763, 'RANGING': 25215, 'CONSOLIDATING': 1968, 'VOLATILE': 4793}  ambiguous=26501 (total=32739)  short_runs_zeroed=1959
2026-05-01 09:51:33,333 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:33,664 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:33,830 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:33,961 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:34,177 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:51:34,197 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:51:34,214 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:51:34,222 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:51:34,223 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:34,325 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 21, 'BIAS_DOWN': 67, 'BIAS_NEUTRAL': 8314}  ambiguous=8303 (total=8402)  short_runs_zeroed=39
2026-05-01 09:51:34,327 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:34,626 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 474, 'RANGING': 25095, 'CONSOLIDATING': 1967, 'VOLATILE': 5203}  ambiguous=26572 (total=32739)  short_runs_zeroed=2307
2026-05-01 09:51:34,639 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:34,965 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:35,125 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:35,254 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:35,489 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:35,510 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:35,527 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:35,538 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:35,539 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:35,638 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 37, 'BIAS_DOWN': 38, 'BIAS_NEUTRAL': 8327}  ambiguous=8321 (total=8402)  short_runs_zeroed=44
2026-05-01 09:51:35,640 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:35,926 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 881, 'RANGING': 25133, 'CONSOLIDATING': 1987, 'VOLATILE': 4739}  ambiguous=26356 (total=32740)  short_runs_zeroed=1985
2026-05-01 09:51:35,945 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:36,292 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:36,463 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:36,611 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:36,850 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:36,873 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:36,890 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:36,899 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:36,900 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:37,001 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 79, 'BIAS_DOWN': 33, 'BIAS_NEUTRAL': 8290}  ambiguous=8328 (total=8402)  short_runs_zeroed=61
2026-05-01 09:51:37,003 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:37,291 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 699, 'RANGING': 24989, 'CONSOLIDATING': 2130, 'VOLATILE': 4923}  ambiguous=26228 (total=32741)  short_runs_zeroed=2004
2026-05-01 09:51:37,308 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:37,649 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:37,811 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:37,942 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:38,173 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:38,197 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:38,215 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:38,224 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:51:38,225 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:38,326 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 42, 'BIAS_DOWN': 44, 'BIAS_NEUTRAL': 8317}  ambiguous=8318 (total=8403)  short_runs_zeroed=44
2026-05-01 09:51:38,328 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:38,622 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 808, 'RANGING': 24290, 'CONSOLIDATING': 2180, 'VOLATILE': 5465}  ambiguous=25645 (total=32743)  short_runs_zeroed=2211
2026-05-01 09:51:38,639 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:38,977 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:39,145 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:39,280 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:51:39,630 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:51:39,658 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:51:39,677 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:51:39,689 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:51:39,690 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:51:39,881 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 107, 'BIAS_DOWN': 115, 'BIAS_NEUTRAL': 19595}  ambiguous=19636 (total=19817)  short_runs_zeroed=112
2026-05-01 09:51:39,884 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:51:40,501 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 1778, 'RANGING': 57285, 'CONSOLIDATING': 4542, 'VOLATILE': 11019}  ambiguous=60314 (total=74624)  short_runs_zeroed=4774
2026-05-01 09:51:40,552 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:51:41,229 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:51:41,485 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:51:41,652 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:51:41,804 INFO train_multi: 44 segments, ~6069667 total bars
2026-05-01 09:51:41,804 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-05-01 09:51:41,804 INFO train_multi: building combined dataset for TF=ALL (44 segments)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
2026-05-01 09:52:45,606 INFO train_multi TF=ALL: 6068347 sequences across 44 segments
2026-05-01 09:52:45,606 INFO train_multi TF=ALL: estimated peak RAM = 11231 MB (train=479964 val=120008 n_feat=78 seq_len=30)
2026-05-01 09:52:47,127 INFO train_multi TF=ALL: train=479964 val=120008 (5623 MB tensors)
2026-05-01 09:52:54,453 INFO train_multi TF=ALL: warm-start detected — using CosineAnnealingLR (lr=3e-05, patience=12)
2026-05-01 09:53:12,656 INFO train_multi TF=ALL epoch 1/50 train=0.5795 val=0.6090
2026-05-01 09:53:12,663 INFO WeightsManifest written → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/weights_manifest.json
2026-05-01 09:53:12,663 INFO GRULSTMPredictor saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 09:53:12,664 INFO train_multi TF=ALL: new best val=0.6090 — saved
2026-05-01 09:53:28,578 INFO train_multi TF=ALL epoch 2/50 train=0.5790 val=0.6098
2026-05-01 09:53:43,920 INFO train_multi TF=ALL epoch 3/50 train=0.5790 val=0.6095
2026-05-01 09:53:58,932 INFO train_multi TF=ALL epoch 4/50 train=0.5788 val=0.6097
2026-05-01 09:54:14,293 INFO train_multi TF=ALL epoch 5/50 train=0.5786 val=0.6090
2026-05-01 09:54:29,898 INFO train_multi TF=ALL epoch 6/50 train=0.5781 val=0.6094
2026-05-01 09:54:45,602 INFO train_multi TF=ALL epoch 7/50 train=0.5781 val=0.6094
2026-05-01 09:55:01,583 INFO train_multi TF=ALL epoch 8/50 train=0.5778 val=0.6111
2026-05-01 09:55:17,213 INFO train_multi TF=ALL epoch 9/50 train=0.5779 val=0.6097
2026-05-01 09:55:32,409 INFO train_multi TF=ALL epoch 10/50 train=0.5777 val=0.6109
2026-05-01 09:55:47,759 INFO train_multi TF=ALL epoch 11/50 train=0.5779 val=0.6107
2026-05-01 09:56:03,510 INFO train_multi TF=ALL epoch 12/50 train=0.5772 val=0.6115
2026-05-01 09:56:19,085 INFO train_multi TF=ALL epoch 13/50 train=0.5773 val=0.6107
2026-05-01 09:56:19,086 INFO train_multi TF=ALL early stop at epoch 13
2026-05-01 09:56:19,274 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-05-01 09:56:19,274 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-05-01 09:56:19,275 INFO Retrain complete. Total wall-clock: 294.0s
  DONE  Retrain gru [train-split retrain]
  START Retrain regime [train-split retrain]
2026-05-01 09:56:22,728 INFO retrain environment: KAGGLE
2026-05-01 09:56:24,679 INFO Device: CUDA (2 GPU(s))
2026-05-01 09:56:24,689 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 09:56:24,690 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 09:56:24,690 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-01 09:56:24,690 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-01 09:56:24,690 INFO Retrain data split: train
2026-05-01 09:56:24,692 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-05-01 09:56:24,862 INFO NumExpr defaulting to 4 threads.
2026-05-01 09:56:25,105 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-01 09:56:25,106 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 09:56:25,106 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 09:56:25,106 INFO Regime phase macro_correlations: 0.0s
2026-05-01 09:56:25,106 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-01 09:56:25,182 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-05-01 09:56:25,226 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-05-01 09:56:25,227 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:25,246 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:25,265 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:25,284 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:25,304 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:25,323 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:25,341 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:25,360 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:25,380 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:25,399 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:25,425 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:56:25,569 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:25,617 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:25,640 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:25,640 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:25,649 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:25,650 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:26,100 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1756, 'BIAS_DOWN': 1771, 'BIAS_NEUTRAL': 4875}  ambiguous=13 (total=8402) horizon=12
2026-05-01 09:56:26,102 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-05-01 09:56:26,322 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:26,368 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:26,390 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:26,390 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:26,399 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:26,400 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:26,823 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1846, 'BIAS_DOWN': 1663, 'BIAS_NEUTRAL': 4893}  ambiguous=13 (total=8402) horizon=12
2026-05-01 09:56:26,825 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-05-01 09:56:27,009 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:27,050 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:27,072 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:27,073 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:27,082 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:27,083 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:27,479 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1725, 'BIAS_DOWN': 1818, 'BIAS_NEUTRAL': 4859}  ambiguous=13 (total=8402) horizon=12
2026-05-01 09:56:27,481 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-05-01 09:56:27,690 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:27,734 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:27,756 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:27,757 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:27,765 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:27,767 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:28,171 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1852, 'BIAS_DOWN': 1677, 'BIAS_NEUTRAL': 4873}  ambiguous=13 (total=8402) horizon=12
2026-05-01 09:56:28,172 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-05-01 09:56:28,366 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:28,409 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:28,434 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:28,435 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:28,444 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:28,445 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:28,837 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1717, 'BIAS_DOWN': 1858, 'BIAS_NEUTRAL': 4828}  ambiguous=13 (total=8403) horizon=12
2026-05-01 09:56:28,839 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-05-01 09:56:29,021 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:29,062 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:29,084 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:29,084 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:29,093 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:29,094 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:29,507 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1823, 'BIAS_DOWN': 1695, 'BIAS_NEUTRAL': 4885}  ambiguous=13 (total=8403) horizon=12
2026-05-01 09:56:29,509 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-05-01 09:56:29,671 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:56:29,704 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:56:29,725 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:56:29,726 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:56:29,734 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:56:29,735 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:30,134 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1837, 'BIAS_DOWN': 1733, 'BIAS_NEUTRAL': 4832}  ambiguous=13 (total=8402) horizon=12
2026-05-01 09:56:30,136 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-05-01 09:56:30,322 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:30,361 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:30,383 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:30,383 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:30,392 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:30,394 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:30,789 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1769, 'BIAS_DOWN': 1779, 'BIAS_NEUTRAL': 4854}  ambiguous=13 (total=8402) horizon=12
2026-05-01 09:56:30,790 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-05-01 09:56:30,973 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:31,016 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:31,038 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:31,038 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:31,047 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:31,048 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:31,449 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1642, 'BIAS_DOWN': 1875, 'BIAS_NEUTRAL': 4885}  ambiguous=13 (total=8402) horizon=12
2026-05-01 09:56:31,451 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-05-01 09:56:31,638 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:31,677 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:31,700 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:31,701 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:31,710 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:31,711 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:32,109 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1651, 'BIAS_DOWN': 1907, 'BIAS_NEUTRAL': 4845}  ambiguous=13 (total=8403) horizon=12
2026-05-01 09:56:32,110 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-05-01 09:56:32,396 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:56:32,464 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:56:32,492 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:56:32,493 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:56:32,505 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:56:32,507 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:56:33,333 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 4279, 'BIAS_DOWN': 4164, 'BIAS_NEUTRAL': 11374}  ambiguous=13 (total=19817) horizon=12
2026-05-01 09:56:33,335 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-05-01 09:56:33,523 INFO Regime phase HTF dataset build: 8.3s (103290 samples)
2026-05-01 09:56:33,524 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260501_095633
2026-05-01 09:56:33,733 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-05-01 09:56:33,735 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=103158 dropped=132 classes={'BIAS_UP': 21809, 'BIAS_DOWN': 21800, 'BIAS_NEUTRAL': 59549})
2026-05-01 09:56:33,763 INFO RegimeClassifier[mode=htf_bias]: 103158 samples, classes={'BIAS_UP': 21809, 'BIAS_DOWN': 21800, 'BIAS_NEUTRAL': 59549}, device=cuda
2026-05-01 09:56:33,764 INFO RegimeClassifier: sample weights — mean=0.787  ambiguous(<0.4)=0.0%
2026-05-01 09:56:33,764 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-05-01 09:56:33,764 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-01 09:56:36,304 INFO Regime epoch  1/50 — tr=1.1647 va=1.1089 acc=0.368 per_class={'BIAS_UP': 0.384, 'BIAS_DOWN': 0.23, 'BIAS_NEUTRAL': 0.412}
2026-05-01 09:56:36,516 INFO Regime epoch  2/50 — tr=1.1654 va=1.1089 acc=0.362
2026-05-01 09:56:36,751 INFO Regime epoch  3/50 — tr=1.1656 va=1.1089 acc=0.361
2026-05-01 09:56:36,948 INFO Regime epoch  4/50 — tr=1.1652 va=1.1088 acc=0.365
2026-05-01 09:56:37,164 INFO Regime epoch  5/50 — tr=1.1656 va=1.1090 acc=0.363 per_class={'BIAS_UP': 0.382, 'BIAS_DOWN': 0.243, 'BIAS_NEUTRAL': 0.401}
2026-05-01 09:56:37,369 INFO Regime epoch  6/50 — tr=1.1649 va=1.1092 acc=0.363
2026-05-01 09:56:37,566 INFO Regime epoch  7/50 — tr=1.1648 va=1.1090 acc=0.361
2026-05-01 09:56:37,759 INFO Regime epoch  8/50 — tr=1.1648 va=1.1092 acc=0.360
2026-05-01 09:56:37,955 INFO Regime epoch  9/50 — tr=1.1655 va=1.1091 acc=0.360
2026-05-01 09:56:38,174 INFO Regime epoch 10/50 — tr=1.1647 va=1.1095 acc=0.354 per_class={'BIAS_UP': 0.392, 'BIAS_DOWN': 0.248, 'BIAS_NEUTRAL': 0.379}
2026-05-01 09:56:38,377 INFO Regime epoch 11/50 — tr=1.1648 va=1.1094 acc=0.356
2026-05-01 09:56:38,575 INFO Regime epoch 12/50 — tr=1.1650 va=1.1095 acc=0.355
2026-05-01 09:56:38,768 INFO Regime epoch 13/50 — tr=1.1637 va=1.1094 acc=0.355
2026-05-01 09:56:38,967 INFO Regime epoch 14/50 — tr=1.1645 va=1.1094 acc=0.352
2026-05-01 09:56:38,967 INFO Regime early stop at epoch 14 (no_improve=10)
2026-05-01 09:56:38,986 WARNING RegimeClassifier accuracy 0.365 < warning floor 0.483 (harder structural labels; check blind backtest economics)
2026-05-01 09:56:38,989 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-01 09:56:38,990 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-01 09:56:38,991 INFO Regime phase HTF train: 5.3s
2026-05-01 09:56:39,121 INFO Regime HTF complete: acc=0.365, n=103290 per_class={'BIAS_UP': 0.379, 'BIAS_DOWN': 0.243, 'BIAS_NEUTRAL': 0.405}
2026-05-01 09:56:39,123 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:56:39,161 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 4279, 'BIAS_DOWN': 4164, 'BIAS_NEUTRAL': 11374}  ambiguous=13 (total=19817) horizon=12
2026-05-01 09:56:39,164 INFO Regime[4H mode=htf_bias] persistence (avg bars/run) on XAUUSD 4H:
{'BIAS_UP': 5.34875, 'BIAS_DOWN': 5.407792207792208, 'BIAS_NEUTRAL': 7.433986928104575}
2026-05-01 09:56:39,170 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (all labels):
{'BIAS_UP': {'n': 4279, 'mean': 0.0017182910764822602, 'mean_over_std': 0.3954951269266541}, 'BIAS_DOWN': {'n': 4164, 'mean': -0.001754885052740925, 'mean_over_std': -0.3973602040268371}, 'BIAS_NEUTRAL': {'n': 11373, 'mean': 6.863221972092698e-05, 'mean_over_std': 0.021025233297328444}}
2026-05-01 09:56:39,170 INFO Regime[4H mode=htf_bias] return separation on XAUUSD 4H (clean labels conf>=0.40):
{'BIAS_UP': {'n': 4279, 'mean': 0.0017182910764822602, 'mean_over_std': 0.3954951269266541}, 'BIAS_DOWN': {'n': 4164, 'mean': -0.001754885052740925, 'mean_over_std': -0.3973602040268371}, 'BIAS_NEUTRAL': {'n': 11361, 'mean': 6.848051916093716e-05, 'mean_over_std': 0.020969591280060183}}
2026-05-01 09:56:39,180 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-05-01 09:56:39,182 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:39,184 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:39,186 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:39,188 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:39,190 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:39,192 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:39,193 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:39,195 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:39,197 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:39,198 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:39,202 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:56:39,213 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:39,215 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:39,216 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:39,216 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:39,216 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:39,218 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:39,754 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8942, 'RANGING': 13380, 'CONSOLIDATING': 8163, 'VOLATILE': 2253}  ambiguous=13 (total=32738) horizon=12
2026-05-01 09:56:39,757 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-05-01 09:56:39,894 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:39,896 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:39,897 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:39,898 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:39,898 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:39,900 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:40,404 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 7969, 'RANGING': 13888, 'CONSOLIDATING': 8178, 'VOLATILE': 2703}  ambiguous=13 (total=32738) horizon=12
2026-05-01 09:56:40,408 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-05-01 09:56:40,552 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:40,554 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:40,555 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:40,556 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:40,556 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:40,558 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:41,066 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8757, 'RANGING': 13514, 'CONSOLIDATING': 8167, 'VOLATILE': 2302}  ambiguous=13 (total=32740) horizon=12
2026-05-01 09:56:41,070 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-05-01 09:56:41,213 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:41,215 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:41,216 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:41,216 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:41,217 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:41,219 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:41,723 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8274, 'RANGING': 13643, 'CONSOLIDATING': 8178, 'VOLATILE': 2644}  ambiguous=13 (total=32739) horizon=12
2026-05-01 09:56:41,727 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-05-01 09:56:41,865 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:41,867 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:41,868 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:41,869 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:41,869 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:41,871 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:42,374 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8713, 'RANGING': 13564, 'CONSOLIDATING': 8168, 'VOLATILE': 2295}  ambiguous=13 (total=32740) horizon=12
2026-05-01 09:56:42,377 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-05-01 09:56:42,515 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:42,518 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:42,519 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:42,519 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:42,519 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:42,522 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:43,023 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8295, 'RANGING': 13768, 'CONSOLIDATING': 8178, 'VOLATILE': 2498}  ambiguous=13 (total=32739) horizon=12
2026-05-01 09:56:43,026 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-05-01 09:56:43,168 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:56:43,170 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:56:43,170 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:56:43,171 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:56:43,171 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-01 09:56:43,173 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:43,674 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 9045, 'RANGING': 13346, 'CONSOLIDATING': 8165, 'VOLATILE': 2183}  ambiguous=13 (total=32739) horizon=12
2026-05-01 09:56:43,677 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-05-01 09:56:43,819 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:43,822 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:43,823 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:43,823 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:43,824 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:43,826 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:44,331 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8133, 'RANGING': 13841, 'CONSOLIDATING': 8176, 'VOLATILE': 2590}  ambiguous=13 (total=32740) horizon=12
2026-05-01 09:56:44,335 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-05-01 09:56:44,477 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:44,480 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:44,481 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:44,481 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:44,482 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:44,484 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:44,997 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 7969, 'RANGING': 13883, 'CONSOLIDATING': 8177, 'VOLATILE': 2712}  ambiguous=13 (total=32741) horizon=12
2026-05-01 09:56:45,000 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-05-01 09:56:45,140 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:45,142 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:45,143 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:45,144 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:45,144 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 09:56:45,146 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 09:56:45,659 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8927, 'RANGING': 13395, 'CONSOLIDATING': 8160, 'VOLATILE': 2261}  ambiguous=13 (total=32743) horizon=12
2026-05-01 09:56:45,662 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-05-01 09:56:45,811 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:56:45,817 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:56:45,819 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:56:45,819 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:56:45,820 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-01 09:56:45,824 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:56:46,951 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 19458, 'RANGING': 31056, 'CONSOLIDATING': 18647, 'VOLATILE': 5463}  ambiguous=13 (total=74624) horizon=12
2026-05-01 09:56:46,959 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-05-01 09:56:47,274 INFO Regime phase LTF dataset build: 8.1s (401471 samples)
2026-05-01 09:56:47,275 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260501_095647
2026-05-01 09:56:47,280 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-05-01 09:56:47,284 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=401339 dropped=132 classes={'TRENDING': 104312, 'RANGING': 166920, 'CONSOLIDATING': 100239, 'VOLATILE': 29868})
2026-05-01 09:56:47,391 INFO RegimeClassifier[mode=ltf_behaviour]: 401339 samples, classes={'TRENDING': 104312, 'RANGING': 166920, 'CONSOLIDATING': 100239, 'VOLATILE': 29868}, device=cuda
2026-05-01 09:56:47,393 INFO RegimeClassifier: sample weights — mean=0.790  ambiguous(<0.4)=0.0%
2026-05-01 09:56:47,393 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-05-01 09:56:47,393 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-01 09:56:48,256 INFO Regime epoch  1/50 — tr=1.2598 va=1.2099 acc=0.373 per_class={'TRENDING': 0.201, 'RANGING': 0.241, 'CONSOLIDATING': 0.694, 'VOLATILE': 0.645}
2026-05-01 09:56:49,021 INFO Regime epoch  2/50 — tr=1.2595 va=1.2104 acc=0.376
2026-05-01 09:56:49,777 INFO Regime epoch  3/50 — tr=1.2592 va=1.2105 acc=0.377
2026-05-01 09:56:50,530 INFO Regime epoch  4/50 — tr=1.2595 va=1.2105 acc=0.378
2026-05-01 09:56:51,358 INFO Regime epoch  5/50 — tr=1.2590 va=1.2104 acc=0.375 per_class={'TRENDING': 0.199, 'RANGING': 0.246, 'CONSOLIDATING': 0.696, 'VOLATILE': 0.639}
2026-05-01 09:56:52,113 INFO Regime epoch  6/50 — tr=1.2598 va=1.2105 acc=0.375
2026-05-01 09:56:52,839 INFO Regime epoch  7/50 — tr=1.2585 va=1.2112 acc=0.377
2026-05-01 09:56:53,608 INFO Regime epoch  8/50 — tr=1.2591 va=1.2108 acc=0.375
2026-05-01 09:56:54,316 INFO Regime epoch  9/50 — tr=1.2592 va=1.2093 acc=0.374
2026-05-01 09:56:55,135 INFO Regime epoch 10/50 — tr=1.2594 va=1.2101 acc=0.377 per_class={'TRENDING': 0.201, 'RANGING': 0.248, 'CONSOLIDATING': 0.701, 'VOLATILE': 0.629}
2026-05-01 09:56:55,898 INFO Regime epoch 11/50 — tr=1.2589 va=1.2088 acc=0.372
2026-05-01 09:56:56,703 INFO Regime epoch 12/50 — tr=1.2577 va=1.2104 acc=0.377
2026-05-01 09:56:57,476 INFO Regime epoch 13/50 — tr=1.2577 va=1.2094 acc=0.372
2026-05-01 09:56:58,231 INFO Regime epoch 14/50 — tr=1.2578 va=1.2097 acc=0.378
2026-05-01 09:56:59,049 INFO Regime epoch 15/50 — tr=1.2580 va=1.2085 acc=0.371 per_class={'TRENDING': 0.215, 'RANGING': 0.229, 'CONSOLIDATING': 0.686, 'VOLATILE': 0.654}
2026-05-01 09:56:59,794 INFO Regime epoch 16/50 — tr=1.2577 va=1.2088 acc=0.377
2026-05-01 09:57:00,543 INFO Regime epoch 17/50 — tr=1.2580 va=1.2087 acc=0.375
2026-05-01 09:57:01,296 INFO Regime epoch 18/50 — tr=1.2580 va=1.2089 acc=0.377
2026-05-01 09:57:02,046 INFO Regime epoch 19/50 — tr=1.2581 va=1.2088 acc=0.377
2026-05-01 09:57:02,833 INFO Regime epoch 20/50 — tr=1.2580 va=1.2085 acc=0.377 per_class={'TRENDING': 0.215, 'RANGING': 0.241, 'CONSOLIDATING': 0.698, 'VOLATILE': 0.628}
2026-05-01 09:57:03,566 INFO Regime epoch 21/50 — tr=1.2579 va=1.2074 acc=0.373
2026-05-01 09:57:04,293 INFO Regime epoch 22/50 — tr=1.2580 va=1.2078 acc=0.373
2026-05-01 09:57:05,016 INFO Regime epoch 23/50 — tr=1.2582 va=1.2080 acc=0.373
2026-05-01 09:57:05,777 INFO Regime epoch 24/50 — tr=1.2581 va=1.2075 acc=0.372
2026-05-01 09:57:06,588 INFO Regime epoch 25/50 — tr=1.2578 va=1.2084 acc=0.374 per_class={'TRENDING': 0.214, 'RANGING': 0.236, 'CONSOLIDATING': 0.694, 'VOLATILE': 0.637}
2026-05-01 09:57:07,324 INFO Regime epoch 26/50 — tr=1.2569 va=1.2082 acc=0.378
2026-05-01 09:57:08,060 INFO Regime epoch 27/50 — tr=1.2574 va=1.2075 acc=0.376
2026-05-01 09:57:08,816 INFO Regime epoch 28/50 — tr=1.2581 va=1.2086 acc=0.378
2026-05-01 09:57:09,552 INFO Regime epoch 29/50 — tr=1.2574 va=1.2083 acc=0.375
2026-05-01 09:57:10,369 INFO Regime epoch 30/50 — tr=1.2579 va=1.2087 acc=0.375 per_class={'TRENDING': 0.221, 'RANGING': 0.235, 'CONSOLIDATING': 0.695, 'VOLATILE': 0.63}
2026-05-01 09:57:11,151 INFO Regime epoch 31/50 — tr=1.2575 va=1.2078 acc=0.371
2026-05-01 09:57:11,151 INFO Regime early stop at epoch 31 (no_improve=10)
2026-05-01 09:57:11,204 WARNING RegimeClassifier accuracy 0.373 < warning floor 0.400 (harder structural labels; check blind backtest economics)
2026-05-01 09:57:11,208 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-01 09:57:11,208 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-01 09:57:11,211 INFO Regime phase LTF train: 23.9s
2026-05-01 09:57:11,342 INFO Regime LTF complete: acc=0.373, n=401471 per_class={'TRENDING': 0.215, 'RANGING': 0.232, 'CONSOLIDATING': 0.695, 'VOLATILE': 0.638}
2026-05-01 09:57:11,346 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 09:57:11,444 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 19458, 'RANGING': 31056, 'CONSOLIDATING': 18647, 'VOLATILE': 5463}  ambiguous=13 (total=74624) horizon=12
2026-05-01 09:57:11,451 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 4.125079499682001, 'RANGING': 4.718322698268004, 'CONSOLIDATING': 5.990041760359782, 'VOLATILE': 3.723926380368098}
2026-05-01 09:57:11,460 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 19458, 'mean': 5.273978151244844e-05, 'mean_over_std': 0.021533559889529843}, 'RANGING': {'n': 31055, 'mean': -2.3091416023246664e-05, 'mean_over_std': -0.012142968527728414}, 'CONSOLIDATING': {'n': 18647, 'mean': 6.38696766285663e-05, 'mean_over_std': 0.041981339266451534}, 'VOLATILE': {'n': 5463, 'mean': -0.00012433839470805868, 'mean_over_std': -0.04867063159870624}}
2026-05-01 09:57:11,461 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 19458, 'mean': 5.273978151244844e-05, 'mean_over_std': 0.021533559889529843}, 'RANGING': {'n': 31043, 'mean': -2.2961083650504332e-05, 'mean_over_std': -0.01207348194887135}, 'CONSOLIDATING': {'n': 18647, 'mean': 6.38696766285663e-05, 'mean_over_std': 0.041981339266451534}, 'VOLATILE': {'n': 5463, 'mean': -0.00012433839470805868, 'mean_over_std': -0.04867063159870624}}
2026-05-01 09:57:11,471 INFO Regime retrain total: 46.8s (504761 samples)
2026-05-01 09:57:11,481 INFO GRU: 2 CUDA device(s) available — using GPU
2026-05-01 09:57:11,481 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 09:57:11,481 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 09:57:11,481 INFO === VectorStore: building similarity indices (parallel feature build) ===
2026-05-01 09:57:11,482 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-05-01 09:57:11,482 INFO Retrain complete. Total wall-clock: 46.8s
  DONE  Retrain regime [train-split retrain]
  SKIP  Quality/RL incremental retrain — clean train-only weights retained

=== Round 3: Backtest on last 3yr (post-retrain evaluation) ===
  START Round 3 - Post-retrain backtest (last 3yr)
2026-05-01 09:57:13,209 INFO === STEP 6: BACKTEST (round3) ===
2026-05-01 09:57:13,211 INFO BT_WINDOW=round3 — post-retrain eval: 2022-08-05 → 2025-08-05 (last 3yr)
2026-05-01 09:57:13,211 INFO ================================================================
  ROUND 3 / 3
================================================================
2026-05-01 09:57:13,211 INFO Round 3 — running backtest: 2022-08-05 → 2025-08-05 (ml_trader, shared ML cache)
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
/kaggle/working/Multi-Bot/trading-system/trading-engine/scripts/../services/feature_engine.py:987: FutureWarning: Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  out["volume"].clip(lower=1e-9).rolling(20, min_periods=1).mean()
2026-05-01 10:01:03,305 WARNING ml_trader: portfolio drawdown 100.4% after trade exit — halting all trading
2026-05-01 10:01:04,897 INFO Round 3 backtest — 196 trades | avg WR=13.8% | avg PF=0.41 | avg Sharpe=-6.46
2026-05-01 10:01:04,897 INFO   ml_trader: 196 trades | WR=13.8% | fixed PF=0.41 | Return=-100.4% | ExpR=-0.512 | DD=100.4% | Sharpe=-6.46
INFO  Loading: /kaggle/working/Multi-Bot/trading-system/backtesting/results/backtest_round_3.json
INFO  Total trades: 196
INFO  Diagnostics CSV → /kaggle/working/Multi-Bot/trading-system/trading-engine/logs/backtest_diagnostics.csv (196 rows)
2026-05-01 10:01:05,244 INFO Round 3: wrote 196 journal entries (total in file: 1184)
  DONE  Round 3 - Post-retrain backtest (last 3yr)
  Saved Round 3 result → round3_summary.json
  Journal after Round 3: 1184 entries

  SKIP  Round 3 Quality+RL retrain — evaluation journals not used for fitting

======================================================================
  BLIND BACKTEST PIPELINE COMPLETE
======================================================================
  Round 1 (val window)          trades=336  WR=20.5%  PF=0.625  Sharpe=-3.289
  Round 2 (blind test)          trades=652  WR=25.3%  PF=0.872  Sharpe=-0.949
  Round 3 (last 3yr)            trades=196  WR=13.8%  PF=0.407  Sharpe=-6.464

