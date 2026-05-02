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
2026-05-02 00:12:08,117 INFO Loading feature-engineered data...
2026-05-02 00:12:08,762 INFO Loaded 221743 rows, 202 features
2026-05-02 00:12:08,763 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-05-02 00:12:08,766 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-05-02 00:12:08,766 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-05-02 00:12:08,766 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-05-02 00:12:08,766 INFO No leakage confirmed: train < val < test timestamps

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
2026-05-02 00:12:11,234 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-02 00:12:11,234 INFO --- Training regime ---
2026-05-02 00:12:11,234 INFO Running retrain --model regime
2026-05-02 00:12:11,418 INFO retrain environment: KAGGLE
2026-05-02 00:12:13,058 INFO Device: CUDA (2 GPU(s))
2026-05-02 00:12:13,069 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-02 00:12:13,069 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-02 00:12:13,069 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-02 00:12:13,071 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-02 00:12:13,071 INFO Retrain data split: train
2026-05-02 00:12:13,072 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-05-02 00:12:13,231 INFO NumExpr defaulting to 4 threads.
2026-05-02 00:12:13,464 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-02 00:12:13,465 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-02 00:12:13,465 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-02 00:12:13,465 INFO Regime phase macro_correlations: 0.0s
2026-05-02 00:12:13,465 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-02 00:12:13,523 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-05-02 00:12:13,561 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-05-02 00:12:13,562 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:13,577 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:13,592 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:13,607 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:13,622 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:13,637 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:13,651 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:13,666 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:13,681 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:13,696 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:13,714 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-02 00:12:13,835 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:13,878 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:13,897 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:13,897 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:13,905 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:13,905 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:14,294 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1756, 'BIAS_DOWN': 1771, 'BIAS_NEUTRAL': 4875}  ambiguous=13 (total=8402) horizon=12
2026-05-02 00:12:14,296 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-05-02 00:12:14,450 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:14,487 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:14,506 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:14,506 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:14,514 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:14,515 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:14,864 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1846, 'BIAS_DOWN': 1663, 'BIAS_NEUTRAL': 4893}  ambiguous=13 (total=8402) horizon=12
2026-05-02 00:12:14,866 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-05-02 00:12:15,044 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:15,081 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:15,100 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:15,101 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:15,108 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:15,109 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:15,474 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1725, 'BIAS_DOWN': 1818, 'BIAS_NEUTRAL': 4859}  ambiguous=13 (total=8402) horizon=12
2026-05-02 00:12:15,476 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-05-02 00:12:15,630 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:15,667 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:15,686 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:15,687 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:15,694 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:15,695 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:16,030 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1852, 'BIAS_DOWN': 1677, 'BIAS_NEUTRAL': 4873}  ambiguous=13 (total=8402) horizon=12
2026-05-02 00:12:16,031 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-05-02 00:12:16,183 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:16,219 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:16,238 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:16,238 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:16,245 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:16,246 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:16,579 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1717, 'BIAS_DOWN': 1858, 'BIAS_NEUTRAL': 4828}  ambiguous=13 (total=8403) horizon=12
2026-05-02 00:12:16,580 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-05-02 00:12:16,721 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:16,754 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:16,772 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:16,772 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:16,780 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:16,781 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:17,114 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1823, 'BIAS_DOWN': 1695, 'BIAS_NEUTRAL': 4885}  ambiguous=13 (total=8403) horizon=12
2026-05-02 00:12:17,115 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-05-02 00:12:17,241 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 00:12:17,269 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-02 00:12:17,286 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 00:12:17,286 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 00:12:17,293 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 00:12:17,294 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:17,640 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1837, 'BIAS_DOWN': 1733, 'BIAS_NEUTRAL': 4832}  ambiguous=13 (total=8402) horizon=12
2026-05-02 00:12:17,642 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-05-02 00:12:17,785 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:17,820 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:17,838 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:17,839 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:17,846 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:17,847 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:18,192 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1769, 'BIAS_DOWN': 1779, 'BIAS_NEUTRAL': 4854}  ambiguous=13 (total=8402) horizon=12
2026-05-02 00:12:18,193 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-05-02 00:12:18,348 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:18,383 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:18,403 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:18,404 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:18,411 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:18,412 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:18,800 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1642, 'BIAS_DOWN': 1875, 'BIAS_NEUTRAL': 4885}  ambiguous=13 (total=8402) horizon=12
2026-05-02 00:12:18,801 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-05-02 00:12:18,952 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:18,986 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:19,006 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:19,006 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:19,014 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:19,015 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:19,356 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1651, 'BIAS_DOWN': 1907, 'BIAS_NEUTRAL': 4845}  ambiguous=13 (total=8403) horizon=12
2026-05-02 00:12:19,357 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-05-02 00:12:19,603 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 00:12:19,664 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-02 00:12:19,688 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 00:12:19,689 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 00:12:19,698 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 00:12:19,700 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-02 00:12:20,446 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 4279, 'BIAS_DOWN': 4164, 'BIAS_NEUTRAL': 11374}  ambiguous=13 (total=19817) horizon=12
2026-05-02 00:12:20,448 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-05-02 00:12:20,580 INFO Regime phase HTF dataset build: 7.1s (103290 samples)
2026-05-02 00:12:20,582 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=103158 dropped=132 classes={'BIAS_UP': 21809, 'BIAS_DOWN': 21800, 'BIAS_NEUTRAL': 59549})
2026-05-02 00:12:20,608 INFO RegimeClassifier[mode=htf_bias]: 103158 samples, classes={'BIAS_UP': 21809, 'BIAS_DOWN': 21800, 'BIAS_NEUTRAL': 59549}, device=cuda
2026-05-02 00:12:20,610 INFO RegimeClassifier[mode=htf_bias]: undersample class BIAS_NEUTRAL: 59549 → 43600
2026-05-02 00:12:20,621 INFO RegimeClassifier[mode=htf_bias]: after undersampling: 87209 samples classes={'BIAS_UP': 21809, 'BIAS_DOWN': 21800, 'BIAS_NEUTRAL': 43600}
2026-05-02 00:12:20,621 INFO RegimeClassifier: sample weights — mean=0.805  ambiguous(<0.4)=0.0%
2026-05-02 00:12:20,912 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-05-02 00:12:20,913 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-02 00:12:25,759 INFO Regime epoch  1/50 — tr=1.2459 va=0.9809 acc=0.450 bal=0.331 per_class={'BIAS_UP': 0.13, 'BIAS_DOWN': 0.038, 'BIAS_NEUTRAL': 0.823}
2026-05-02 00:12:25,904 INFO Regime epoch  2/50 — tr=1.2388 va=0.9824 acc=0.429 bal=0.325
2026-05-02 00:12:26,047 INFO Regime epoch  3/50 — tr=1.2312 va=0.9805 acc=0.410 bal=0.324
2026-05-02 00:12:26,194 INFO Regime epoch  4/50 — tr=1.2232 va=0.9798 acc=0.400 bal=0.322
2026-05-02 00:12:26,337 INFO Regime epoch  5/50 — tr=1.2165 va=0.9790 acc=0.397 bal=0.323 per_class={'BIAS_UP': 0.226, 'BIAS_DOWN': 0.116, 'BIAS_NEUTRAL': 0.628}
2026-05-02 00:12:26,479 INFO Regime epoch  6/50 — tr=1.1955 va=0.9754 acc=0.391 bal=0.325
2026-05-02 00:12:26,631 INFO Regime epoch  7/50 — tr=1.1737 va=0.9723 acc=0.389 bal=0.325
2026-05-02 00:12:26,781 INFO Regime epoch  8/50 — tr=1.1576 va=0.9698 acc=0.392 bal=0.329
2026-05-02 00:12:26,927 INFO Regime epoch  9/50 — tr=1.1408 va=0.9672 acc=0.383 bal=0.329
2026-05-02 00:12:27,071 INFO Regime epoch 10/50 — tr=1.1254 va=0.9656 acc=0.380 bal=0.329 per_class={'BIAS_UP': 0.353, 'BIAS_DOWN': 0.098, 'BIAS_NEUTRAL': 0.536}
2026-05-02 00:12:27,218 INFO Regime epoch 11/50 — tr=1.1114 va=0.9636 acc=0.372 bal=0.329
2026-05-02 00:12:27,218 INFO Regime early stop at epoch 11 (no_improve=10)
2026-05-02 00:12:27,232 INFO Regime phase HTF train: 6.7s
2026-05-02 00:12:27,335 ERROR Regime HTF training failed: Regime validation below acceptance floor: accuracy=0.450 min_overall=0.363 per_class={'BIAS_UP': 0.13, 'BIAS_DOWN': 0.038, 'BIAS_NEUTRAL': 0.823} min_class=0.100 weak_classes=['BIAS_DOWN']. Refusing to save misleading regime weights.
2026-05-02 00:12:27,335 ERROR Regime HTF weights were not created at regime_htf.pkl
2026-05-02 00:12:27,335 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-05-02 00:12:27,338 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:27,340 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:27,341 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:27,343 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:27,344 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:27,346 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:27,347 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:27,349 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:27,351 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:27,352 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:27,355 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-02 00:12:27,363 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:27,366 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:27,367 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:27,367 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:27,367 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:27,369 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:27,847 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8942, 'RANGING': 13380, 'CONSOLIDATING': 8163, 'VOLATILE': 2253}  ambiguous=13 (total=32738) horizon=12
2026-05-02 00:12:27,850 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-05-02 00:12:27,960 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:27,963 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:27,965 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:27,965 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:27,965 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:27,967 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:28,412 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 7969, 'RANGING': 13888, 'CONSOLIDATING': 8178, 'VOLATILE': 2703}  ambiguous=13 (total=32738) horizon=12
2026-05-02 00:12:28,415 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-05-02 00:12:28,523 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:28,526 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:28,527 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:28,527 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:28,527 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:28,529 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:28,967 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8757, 'RANGING': 13514, 'CONSOLIDATING': 8167, 'VOLATILE': 2302}  ambiguous=13 (total=32740) horizon=12
2026-05-02 00:12:28,970 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-05-02 00:12:29,085 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:29,087 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:29,088 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:29,089 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:29,089 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:29,091 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:29,551 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8274, 'RANGING': 13643, 'CONSOLIDATING': 8178, 'VOLATILE': 2644}  ambiguous=13 (total=32739) horizon=12
2026-05-02 00:12:29,554 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-05-02 00:12:29,663 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:29,666 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:29,667 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:29,667 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:29,667 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:29,669 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:30,115 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8713, 'RANGING': 13564, 'CONSOLIDATING': 8168, 'VOLATILE': 2295}  ambiguous=13 (total=32740) horizon=12
2026-05-02 00:12:30,118 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-05-02 00:12:30,233 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:30,235 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:30,236 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:30,236 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:30,237 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:30,239 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:30,717 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8295, 'RANGING': 13768, 'CONSOLIDATING': 8178, 'VOLATILE': 2498}  ambiguous=13 (total=32739) horizon=12
2026-05-02 00:12:30,720 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-05-02 00:12:30,835 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-02 00:12:30,836 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-02 00:12:30,837 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-02 00:12:30,838 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-02 00:12:30,838 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-02 00:12:30,840 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:31,261 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 9045, 'RANGING': 13346, 'CONSOLIDATING': 8165, 'VOLATILE': 2183}  ambiguous=13 (total=32739) horizon=12
2026-05-02 00:12:31,263 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-05-02 00:12:31,376 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:31,378 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:31,379 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:31,380 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:31,380 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:31,382 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:31,805 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8133, 'RANGING': 13841, 'CONSOLIDATING': 8176, 'VOLATILE': 2590}  ambiguous=13 (total=32740) horizon=12
2026-05-02 00:12:31,808 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-05-02 00:12:31,917 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:31,920 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:31,921 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:31,921 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:31,921 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:31,923 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:32,350 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 7969, 'RANGING': 13883, 'CONSOLIDATING': 8177, 'VOLATILE': 2712}  ambiguous=13 (total=32741) horizon=12
2026-05-02 00:12:32,353 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-05-02 00:12:32,464 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:32,466 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:32,467 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:32,468 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:32,468 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-02 00:12:32,470 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-02 00:12:32,897 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8927, 'RANGING': 13395, 'CONSOLIDATING': 8160, 'VOLATILE': 2261}  ambiguous=13 (total=32743) horizon=12
2026-05-02 00:12:32,900 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-05-02 00:12:33,010 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-02 00:12:33,019 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-02 00:12:33,022 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-02 00:12:33,022 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-02 00:12:33,023 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-02 00:12:33,026 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-02 00:12:33,921 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 19458, 'RANGING': 31056, 'CONSOLIDATING': 18647, 'VOLATILE': 5463}  ambiguous=13 (total=74624) horizon=12
2026-05-02 00:12:33,927 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-05-02 00:12:34,167 INFO Regime phase LTF dataset build: 6.8s (401471 samples)
2026-05-02 00:12:34,170 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=401339 dropped=132 classes={'TRENDING': 104312, 'RANGING': 166920, 'CONSOLIDATING': 100239, 'VOLATILE': 29868})
2026-05-02 00:12:34,257 INFO RegimeClassifier[mode=ltf_behaviour]: 401339 samples, classes={'TRENDING': 104312, 'RANGING': 166920, 'CONSOLIDATING': 100239, 'VOLATILE': 29868}, device=cuda
2026-05-02 00:12:34,260 INFO RegimeClassifier[mode=ltf_behaviour]: undersample class TRENDING: 104312 → 59736
2026-05-02 00:12:34,262 INFO RegimeClassifier[mode=ltf_behaviour]: undersample class RANGING: 166920 → 59736
2026-05-02 00:12:34,264 INFO RegimeClassifier[mode=ltf_behaviour]: undersample class CONSOLIDATING: 100239 → 59736
2026-05-02 00:12:34,276 INFO RegimeClassifier[mode=ltf_behaviour]: after undersampling: 209076 samples classes={'TRENDING': 59736, 'RANGING': 59736, 'CONSOLIDATING': 59736, 'VOLATILE': 29868}
2026-05-02 00:12:34,276 INFO RegimeClassifier: sample weights — mean=0.794  ambiguous(<0.4)=0.0%
2026-05-02 00:12:34,278 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-05-02 00:12:34,279 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-02 00:12:34,630 INFO Regime epoch  1/50 — tr=1.3677 va=1.2029 acc=0.295 bal=0.280 per_class={'TRENDING': 0.063, 'RANGING': 0.336, 'CONSOLIDATING': 0.544, 'VOLATILE': 0.176}
2026-05-02 00:12:34,986 INFO Regime epoch  2/50 — tr=1.3601 va=1.1918 acc=0.316 bal=0.310
2026-05-02 00:12:35,349 INFO Regime epoch  3/50 — tr=1.3479 va=1.1811 acc=0.331 bal=0.335
2026-05-02 00:12:35,676 INFO Regime epoch  4/50 — tr=1.3255 va=1.1633 acc=0.354 bal=0.361
2026-05-02 00:12:36,008 INFO Regime epoch  5/50 — tr=1.3027 va=1.1468 acc=0.367 bal=0.377 per_class={'TRENDING': 0.034, 'RANGING': 0.314, 'CONSOLIDATING': 0.712, 'VOLATILE': 0.45}
2026-05-02 00:12:36,350 INFO Regime epoch  6/50 — tr=1.2835 va=1.1331 acc=0.376 bal=0.390
2026-05-02 00:12:36,680 INFO Regime epoch  7/50 — tr=1.2670 va=1.1216 acc=0.381 bal=0.399
2026-05-02 00:12:37,017 INFO Regime epoch  8/50 — tr=1.2512 va=1.1148 acc=0.384 bal=0.404
2026-05-02 00:12:37,352 INFO Regime epoch  9/50 — tr=1.2366 va=1.1089 acc=0.388 bal=0.409
2026-05-02 00:12:37,679 INFO Regime epoch 10/50 — tr=1.2240 va=1.1050 acc=0.390 bal=0.413 per_class={'TRENDING': 0.045, 'RANGING': 0.334, 'CONSOLIDATING': 0.705, 'VOLATILE': 0.567}
2026-05-02 00:12:38,033 INFO Regime epoch 11/50 — tr=1.2147 va=1.1022 acc=0.391 bal=0.417
2026-05-02 00:12:38,367 INFO Regime epoch 12/50 — tr=1.2060 va=1.1004 acc=0.389 bal=0.417
2026-05-02 00:12:38,691 INFO Regime epoch 13/50 — tr=1.1970 va=1.0993 acc=0.391 bal=0.419
2026-05-02 00:12:39,032 INFO Regime epoch 14/50 — tr=1.1901 va=1.0991 acc=0.393 bal=0.422
2026-05-02 00:12:39,377 INFO Regime epoch 15/50 — tr=1.1827 va=1.0976 acc=0.392 bal=0.424 per_class={'TRENDING': 0.039, 'RANGING': 0.311, 'CONSOLIDATING': 0.707, 'VOLATILE': 0.639}
2026-05-02 00:12:39,730 INFO Regime epoch 16/50 — tr=1.1774 va=1.0967 acc=0.392 bal=0.425
2026-05-02 00:12:40,054 INFO Regime epoch 17/50 — tr=1.1733 va=1.0971 acc=0.391 bal=0.426
2026-05-02 00:12:40,402 INFO Regime epoch 18/50 — tr=1.1692 va=1.0968 acc=0.391 bal=0.427
2026-05-02 00:12:40,758 INFO Regime epoch 19/50 — tr=1.1657 va=1.0961 acc=0.392 bal=0.429
2026-05-02 00:12:41,082 INFO Regime epoch 20/50 — tr=1.1632 va=1.0965 acc=0.392 bal=0.429 per_class={'TRENDING': 0.028, 'RANGING': 0.294, 'CONSOLIDATING': 0.716, 'VOLATILE': 0.681}
2026-05-02 00:12:41,407 INFO Regime epoch 21/50 — tr=1.1587 va=1.0956 acc=0.392 bal=0.429
2026-05-02 00:12:41,728 INFO Regime epoch 22/50 — tr=1.1573 va=1.0957 acc=0.391 bal=0.430
2026-05-02 00:12:42,055 INFO Regime epoch 23/50 — tr=1.1548 va=1.0951 acc=0.391 bal=0.430
2026-05-02 00:12:42,388 INFO Regime epoch 24/50 — tr=1.1533 va=1.0952 acc=0.391 bal=0.430
2026-05-02 00:12:42,725 INFO Regime epoch 25/50 — tr=1.1498 va=1.0948 acc=0.389 bal=0.430 per_class={'TRENDING': 0.023, 'RANGING': 0.266, 'CONSOLIDATING': 0.727, 'VOLATILE': 0.703}
2026-05-02 00:12:43,066 INFO Regime epoch 26/50 — tr=1.1500 va=1.0949 acc=0.389 bal=0.430
2026-05-02 00:12:43,418 INFO Regime epoch 27/50 — tr=1.1469 va=1.0945 acc=0.388 bal=0.429
2026-05-02 00:12:43,757 INFO Regime epoch 28/50 — tr=1.1463 va=1.0939 acc=0.388 bal=0.429
2026-05-02 00:12:44,089 INFO Regime epoch 29/50 — tr=1.1446 va=1.0938 acc=0.387 bal=0.430
2026-05-02 00:12:44,436 INFO Regime epoch 30/50 — tr=1.1451 va=1.0935 acc=0.388 bal=0.430 per_class={'TRENDING': 0.021, 'RANGING': 0.243, 'CONSOLIDATING': 0.743, 'VOLATILE': 0.713}
2026-05-02 00:12:44,757 INFO Regime epoch 31/50 — tr=1.1423 va=1.0933 acc=0.388 bal=0.430
2026-05-02 00:12:45,098 INFO Regime epoch 32/50 — tr=1.1427 va=1.0933 acc=0.388 bal=0.430
2026-05-02 00:12:45,452 INFO Regime epoch 33/50 — tr=1.1416 va=1.0936 acc=0.389 bal=0.431
2026-05-02 00:12:45,769 INFO Regime epoch 34/50 — tr=1.1415 va=1.0929 acc=0.388 bal=0.431
2026-05-02 00:12:46,097 INFO Regime epoch 35/50 — tr=1.1400 va=1.0926 acc=0.387 bal=0.431 per_class={'TRENDING': 0.02, 'RANGING': 0.236, 'CONSOLIDATING': 0.74, 'VOLATILE': 0.727}
2026-05-02 00:12:46,434 INFO Regime epoch 36/50 — tr=1.1404 va=1.0926 acc=0.388 bal=0.431
2026-05-02 00:12:46,774 INFO Regime epoch 37/50 — tr=1.1390 va=1.0926 acc=0.387 bal=0.431
2026-05-02 00:12:47,111 INFO Regime epoch 38/50 — tr=1.1386 va=1.0926 acc=0.388 bal=0.431
2026-05-02 00:12:47,440 INFO Regime epoch 39/50 — tr=1.1392 va=1.0922 acc=0.388 bal=0.431
2026-05-02 00:12:47,781 INFO Regime epoch 40/50 — tr=1.1382 va=1.0923 acc=0.388 bal=0.431 per_class={'TRENDING': 0.019, 'RANGING': 0.236, 'CONSOLIDATING': 0.744, 'VOLATILE': 0.725}
2026-05-02 00:12:48,116 INFO Regime epoch 41/50 — tr=1.1385 va=1.0920 acc=0.386 bal=0.431
2026-05-02 00:12:48,455 INFO Regime epoch 42/50 — tr=1.1383 va=1.0923 acc=0.386 bal=0.431
2026-05-02 00:12:48,777 INFO Regime epoch 43/50 — tr=1.1376 va=1.0922 acc=0.388 bal=0.431
2026-05-02 00:12:49,117 INFO Regime epoch 44/50 — tr=1.1374 va=1.0918 acc=0.387 bal=0.431
2026-05-02 00:12:49,458 INFO Regime epoch 45/50 — tr=1.1368 va=1.0919 acc=0.387 bal=0.431 per_class={'TRENDING': 0.019, 'RANGING': 0.231, 'CONSOLIDATING': 0.745, 'VOLATILE': 0.731}
2026-05-02 00:12:49,788 INFO Regime epoch 46/50 — tr=1.1372 va=1.0924 acc=0.387 bal=0.431
2026-05-02 00:12:50,122 INFO Regime epoch 47/50 — tr=1.1375 va=1.0918 acc=0.387 bal=0.431
2026-05-02 00:12:50,476 INFO Regime epoch 48/50 — tr=1.1367 va=1.0917 acc=0.387 bal=0.431
2026-05-02 00:12:50,832 INFO Regime epoch 49/50 — tr=1.1375 va=1.0923 acc=0.387 bal=0.431
2026-05-02 00:12:50,832 INFO Regime early stop at epoch 49 (no_improve=10)
2026-05-02 00:12:50,859 INFO Regime phase LTF train: 16.7s
2026-05-02 00:12:50,967 ERROR Regime LTF training failed: Regime validation below acceptance floor: accuracy=0.388 min_overall=0.280 per_class={'TRENDING': 0.018, 'RANGING': 0.235, 'CONSOLIDATING': 0.742, 'VOLATILE': 0.73} min_class=0.100 weak_classes=['TRENDING']. Refusing to save misleading regime weights.
2026-05-02 00:12:50,967 ERROR Regime LTF weights were not created at regime_ltf.pkl
2026-05-02 00:12:50,967 INFO Regime retrain total: 37.9s (504761 samples)
2026-05-02 00:12:50,977 INFO Retrain complete. Total wall-clock: 37.9s

=== TRAINING COMPLETE ===
  regime: FAILED: exit 1
2026-05-02 00:12:53,330 ERROR retrain regime failed (exit 1)
2026-05-02 00:12:53,331 ERROR Model regime failed: exit 1
2026-05-02 00:12:53,331 WARNING   [MISSING] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-02 00:12:53,331 WARNING   [MISSING] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-02 00:12:53,331 WARNING   [MISSING] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-02 00:12:53,331 INFO   [DEFERRED] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl (expected after Round 1)
2026-05-02 00:12:53,331 INFO   [DEFERRED] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip (expected after Round 1)
2026-05-02 00:12:53,332 WARNING Missing required weights: ['gru_lstm', 'regime_htf', 'regime_ltf'] — run retrain_incremental.py for each
2026-05-02 00:12:53,332 INFO Deferred until post-Round-1 journal retrain: ['quality_scorer', 'rl_ppo']
2026-05-02 00:12:53,333 INFO Saved 20 retrain records to metrics/
2026-05-02 00:12:53,333 ERROR Step 7a failed; required training/artifacts missing: ['gru_lstm', 'regime', 'regime_htf', 'regime_ltf']
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/kaggle/working/Multi-Bot/trading-system/kaggle_train.py in <module>
    296 
    297 print("\n=== Phase 7a: Train GRU + Regime (train set only) ===")
--> 298 run_step(
    299     "Step 7a - GRU+Regime",
    300     "step7_train.py",

/kaggle/working/Multi-Bot/trading-system/kaggle_train.py in run_step(name, script, done_check, extra_env)
    186     )
    187     if result.returncode != 0:
--> 188         raise RuntimeError(f"{name} FAILED (exit {result.returncode})")
    189     print(f"  DONE  {name}")
    190 

RuntimeError: Step 7a - GRU+Regime FAILED (exit 1)
