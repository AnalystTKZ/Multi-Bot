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
2026-05-01 16:51:39,944 INFO Loading feature-engineered data...
2026-05-01 16:51:40,595 INFO Loaded 221743 rows, 202 features
2026-05-01 16:51:40,597 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-05-01 16:51:40,599 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-05-01 16:51:40,599 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-05-01 16:51:40,600 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-05-01 16:51:40,600 INFO No leakage confirmed: train < val < test timestamps

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
2026-05-01 16:51:43,065 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-05-01 16:51:43,065 INFO --- Training regime ---
2026-05-01 16:51:43,066 INFO Running retrain --model regime
2026-05-01 16:51:43,258 INFO retrain environment: KAGGLE
2026-05-01 16:51:45,017 INFO Device: CUDA (2 GPU(s))
2026-05-01 16:51:45,028 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 16:51:45,028 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 16:51:45,029 INFO cuDNN benchmark=True, TF32 matmul=True
2026-05-01 16:51:45,030 INFO PyTorch CPU threads: 4 intra / 2 interop
2026-05-01 16:51:45,030 INFO Retrain data split: train
2026-05-01 16:51:45,031 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-05-01 16:51:45,199 INFO NumExpr defaulting to 4 threads.
2026-05-01 16:51:45,475 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-05-01 16:51:45,475 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-05-01 16:51:45,476 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-05-01 16:51:45,476 INFO Regime phase macro_correlations: 0.0s
2026-05-01 16:51:45,476 INFO Regime: skipping GMM fit; structural forward-path labels are the default target
2026-05-01 16:51:45,551 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-05-01 16:51:45,592 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-05-01 16:51:45,593 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:45,610 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:45,626 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:45,642 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:45,658 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:45,673 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:45,689 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:45,706 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:45,722 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:45,741 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:45,764 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 16:51:45,898 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:45,943 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:45,963 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:45,964 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:45,972 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:45,973 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:46,368 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1756, 'BIAS_DOWN': 1771, 'BIAS_NEUTRAL': 4875}  ambiguous=13 (total=8402) horizon=12
2026-05-01 16:51:46,369 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-05-01 16:51:46,539 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:46,577 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:46,604 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:46,605 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:46,613 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:46,614 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:46,955 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1846, 'BIAS_DOWN': 1663, 'BIAS_NEUTRAL': 4893}  ambiguous=13 (total=8402) horizon=12
2026-05-01 16:51:46,957 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-05-01 16:51:47,137 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:47,178 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:47,198 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:47,199 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:47,206 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:47,207 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:47,566 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1725, 'BIAS_DOWN': 1818, 'BIAS_NEUTRAL': 4859}  ambiguous=13 (total=8402) horizon=12
2026-05-01 16:51:47,567 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-05-01 16:51:47,758 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:47,798 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:47,820 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:47,821 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:47,830 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:47,831 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:48,193 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1852, 'BIAS_DOWN': 1677, 'BIAS_NEUTRAL': 4873}  ambiguous=13 (total=8402) horizon=12
2026-05-01 16:51:48,194 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-05-01 16:51:48,358 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:48,394 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:48,414 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:48,415 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:48,422 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:48,423 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:48,771 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1717, 'BIAS_DOWN': 1858, 'BIAS_NEUTRAL': 4828}  ambiguous=13 (total=8403) horizon=12
2026-05-01 16:51:48,772 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-05-01 16:51:48,925 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:48,959 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:48,980 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:48,980 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:48,988 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:48,989 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:49,334 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1823, 'BIAS_DOWN': 1695, 'BIAS_NEUTRAL': 4885}  ambiguous=13 (total=8403) horizon=12
2026-05-01 16:51:49,336 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-05-01 16:51:49,482 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-01 16:51:49,514 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-01 16:51:49,532 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-01 16:51:49,532 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-01 16:51:49,539 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-01 16:51:49,540 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:49,872 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1837, 'BIAS_DOWN': 1733, 'BIAS_NEUTRAL': 4832}  ambiguous=13 (total=8402) horizon=12
2026-05-01 16:51:49,874 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-05-01 16:51:50,036 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:50,078 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:50,097 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:50,098 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:50,105 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:50,106 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:50,500 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1769, 'BIAS_DOWN': 1779, 'BIAS_NEUTRAL': 4854}  ambiguous=13 (total=8402) horizon=12
2026-05-01 16:51:50,502 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-05-01 16:51:50,673 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:50,707 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:50,725 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:50,725 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:50,733 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:50,734 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:51,085 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1642, 'BIAS_DOWN': 1875, 'BIAS_NEUTRAL': 4885}  ambiguous=13 (total=8402) horizon=12
2026-05-01 16:51:51,087 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-05-01 16:51:51,250 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:51,287 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:51,308 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:51,309 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:51,317 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:51:51,318 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:51:51,661 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 1651, 'BIAS_DOWN': 1907, 'BIAS_NEUTRAL': 4845}  ambiguous=13 (total=8403) horizon=12
2026-05-01 16:51:51,662 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-05-01 16:51:51,929 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-01 16:51:51,996 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-01 16:51:52,023 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-01 16:51:52,024 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-01 16:51:52,036 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-01 16:51:52,037 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-05-01 16:51:52,774 INFO Structural labels HTF_BIAS [4H]: {'BIAS_UP': 4279, 'BIAS_DOWN': 4164, 'BIAS_NEUTRAL': 11374}  ambiguous=13 (total=19817) horizon=12
2026-05-01 16:51:52,776 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-05-01 16:51:52,917 INFO Regime phase HTF dataset build: 7.4s (103290 samples)
2026-05-01 16:51:52,918 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_htf.pkl_20260501_165152
2026-05-01 16:51:53,218 INFO RegimeClassifier[mode=htf_bias] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl (device=cuda, features=34, n_classes=3)
2026-05-01 16:51:53,219 INFO RegimeClassifier[mode=htf_bias]: dropped ambiguous labels below 0.40 (kept=103158 dropped=132 classes={'BIAS_UP': 21809, 'BIAS_DOWN': 21800, 'BIAS_NEUTRAL': 59549})
2026-05-01 16:51:53,241 INFO RegimeClassifier[mode=htf_bias]: 103158 samples, classes={'BIAS_UP': 21809, 'BIAS_DOWN': 21800, 'BIAS_NEUTRAL': 59549}, device=cuda
2026-05-01 16:51:53,242 INFO RegimeClassifier: sample weights — mean=0.787  ambiguous(<0.4)=0.0%
2026-05-01 16:51:53,242 INFO RegimeClassifier[mode=htf_bias]: warm start from existing weights
2026-05-01 16:51:53,242 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-01 16:51:58,109 INFO Regime epoch  1/50 — tr=0.8227 va=0.8242 acc=0.365 per_class={'BIAS_UP': 0.383, 'BIAS_DOWN': 0.236, 'BIAS_NEUTRAL': 0.405}
2026-05-01 16:51:58,280 INFO Regime epoch  2/50 — tr=0.8227 va=0.8240 acc=0.363
2026-05-01 16:51:58,452 INFO Regime epoch  3/50 — tr=0.8205 va=0.8235 acc=0.354
2026-05-01 16:51:58,621 INFO Regime epoch  4/50 — tr=0.8190 va=0.8229 acc=0.351
2026-05-01 16:51:58,808 INFO Regime epoch  5/50 — tr=0.8178 va=0.8222 acc=0.341 per_class={'BIAS_UP': 0.431, 'BIAS_DOWN': 0.24, 'BIAS_NEUTRAL': 0.343}
2026-05-01 16:51:58,980 INFO Regime epoch  6/50 — tr=0.8134 va=0.8210 acc=0.327
2026-05-01 16:51:59,165 INFO Regime epoch  7/50 — tr=0.8108 va=0.8201 acc=0.317
2026-05-01 16:51:59,343 INFO Regime epoch  8/50 — tr=0.8073 va=0.8189 acc=0.304
2026-05-01 16:51:59,516 INFO Regime epoch  9/50 — tr=0.8050 va=0.8181 acc=0.298
2026-05-01 16:51:59,702 INFO Regime epoch 10/50 — tr=0.8020 va=0.8172 acc=0.291 per_class={'BIAS_UP': 0.523, 'BIAS_DOWN': 0.255, 'BIAS_NEUTRAL': 0.214}
2026-05-01 16:51:59,877 INFO Regime epoch 11/50 — tr=0.7991 va=0.8167 acc=0.287
2026-05-01 16:52:00,052 INFO Regime epoch 12/50 — tr=0.7977 va=0.8164 acc=0.283
2026-05-01 16:52:00,241 INFO Regime epoch 13/50 — tr=0.7969 va=0.8158 acc=0.278
2026-05-01 16:52:00,414 INFO Regime epoch 14/50 — tr=0.7962 va=0.8154 acc=0.275
2026-05-01 16:52:00,610 INFO Regime epoch 15/50 — tr=0.7949 va=0.8152 acc=0.272 per_class={'BIAS_UP': 0.605, 'BIAS_DOWN': 0.224, 'BIAS_NEUTRAL': 0.162}
2026-05-01 16:52:00,785 INFO Regime epoch 16/50 — tr=0.7941 va=0.8151 acc=0.270
2026-05-01 16:52:00,962 INFO Regime epoch 17/50 — tr=0.7934 va=0.8151 acc=0.270
2026-05-01 16:52:01,141 INFO Regime epoch 18/50 — tr=0.7926 va=0.8147 acc=0.267
2026-05-01 16:52:01,322 INFO Regime epoch 19/50 — tr=0.7928 va=0.8147 acc=0.269
2026-05-01 16:52:01,519 INFO Regime epoch 20/50 — tr=0.7923 va=0.8144 acc=0.264 per_class={'BIAS_UP': 0.63, 'BIAS_DOWN': 0.211, 'BIAS_NEUTRAL': 0.143}
2026-05-01 16:52:01,701 INFO Regime epoch 21/50 — tr=0.7918 va=0.8142 acc=0.261
2026-05-01 16:52:01,873 INFO Regime epoch 22/50 — tr=0.7913 va=0.8142 acc=0.259
2026-05-01 16:52:02,049 INFO Regime epoch 23/50 — tr=0.7906 va=0.8142 acc=0.259
2026-05-01 16:52:02,236 INFO Regime epoch 24/50 — tr=0.7901 va=0.8141 acc=0.260
2026-05-01 16:52:02,432 INFO Regime epoch 25/50 — tr=0.7902 va=0.8140 acc=0.258 per_class={'BIAS_UP': 0.643, 'BIAS_DOWN': 0.211, 'BIAS_NEUTRAL': 0.127}
2026-05-01 16:52:02,604 INFO Regime epoch 26/50 — tr=0.7897 va=0.8139 acc=0.255
2026-05-01 16:52:02,778 INFO Regime epoch 27/50 — tr=0.7887 va=0.8136 acc=0.251
2026-05-01 16:52:02,955 INFO Regime epoch 28/50 — tr=0.7892 va=0.8136 acc=0.251
2026-05-01 16:52:03,133 INFO Regime epoch 29/50 — tr=0.7892 va=0.8137 acc=0.251
2026-05-01 16:52:03,316 INFO Regime epoch 30/50 — tr=0.7891 va=0.8135 acc=0.248 per_class={'BIAS_UP': 0.659, 'BIAS_DOWN': 0.212, 'BIAS_NEUTRAL': 0.104}
2026-05-01 16:52:03,482 INFO Regime epoch 31/50 — tr=0.7891 va=0.8135 acc=0.246
2026-05-01 16:52:03,649 INFO Regime epoch 32/50 — tr=0.7891 va=0.8134 acc=0.248
2026-05-01 16:52:03,820 INFO Regime epoch 33/50 — tr=0.7888 va=0.8134 acc=0.243
2026-05-01 16:52:04,005 INFO Regime epoch 34/50 — tr=0.7890 va=0.8133 acc=0.242
2026-05-01 16:52:04,201 INFO Regime epoch 35/50 — tr=0.7886 va=0.8133 acc=0.242 per_class={'BIAS_UP': 0.671, 'BIAS_DOWN': 0.209, 'BIAS_NEUTRAL': 0.09}
2026-05-01 16:52:04,373 INFO Regime epoch 36/50 — tr=0.7887 va=0.8134 acc=0.243
2026-05-01 16:52:04,544 INFO Regime epoch 37/50 — tr=0.7886 va=0.8132 acc=0.241
2026-05-01 16:52:04,716 INFO Regime epoch 38/50 — tr=0.7879 va=0.8131 acc=0.237
2026-05-01 16:52:04,888 INFO Regime epoch 39/50 — tr=0.7883 va=0.8131 acc=0.239
2026-05-01 16:52:05,079 INFO Regime epoch 40/50 — tr=0.7880 va=0.8131 acc=0.240 per_class={'BIAS_UP': 0.691, 'BIAS_DOWN': 0.196, 'BIAS_NEUTRAL': 0.084}
2026-05-01 16:52:05,258 INFO Regime epoch 41/50 — tr=0.7882 va=0.8131 acc=0.239
2026-05-01 16:52:05,428 INFO Regime epoch 42/50 — tr=0.7883 va=0.8132 acc=0.240
2026-05-01 16:52:05,601 INFO Regime epoch 43/50 — tr=0.7881 va=0.8132 acc=0.239
2026-05-01 16:52:05,767 INFO Regime epoch 44/50 — tr=0.7879 va=0.8132 acc=0.240
2026-05-01 16:52:05,956 INFO Regime epoch 45/50 — tr=0.7878 va=0.8130 acc=0.238 per_class={'BIAS_UP': 0.688, 'BIAS_DOWN': 0.202, 'BIAS_NEUTRAL': 0.078}
2026-05-01 16:52:06,132 INFO Regime epoch 46/50 — tr=0.7879 va=0.8132 acc=0.240
2026-05-01 16:52:06,315 INFO Regime epoch 47/50 — tr=0.7881 va=0.8130 acc=0.237
2026-05-01 16:52:06,485 INFO Regime epoch 48/50 — tr=0.7883 va=0.8130 acc=0.238
2026-05-01 16:52:06,654 INFO Regime epoch 49/50 — tr=0.7883 va=0.8132 acc=0.239
2026-05-01 16:52:06,847 INFO Regime epoch 50/50 — tr=0.7882 va=0.8131 acc=0.237 per_class={'BIAS_UP': 0.683, 'BIAS_DOWN': 0.206, 'BIAS_NEUTRAL': 0.078}
2026-05-01 16:52:06,862 INFO Regime phase HTF train: 13.6s
2026-05-01 16:52:06,979 ERROR Regime HTF training failed: Regime validation below acceptance floor: accuracy=0.238 min_overall=0.363 per_class={'BIAS_UP': 0.682, 'BIAS_DOWN': 0.206, 'BIAS_NEUTRAL': 0.079} min_class=0.100 weak_classes=['BIAS_NEUTRAL']. Refusing to save misleading regime weights.
2026-05-01 16:52:06,980 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-05-01 16:52:06,982 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:06,984 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:06,985 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:06,987 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:06,989 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:06,990 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:06,992 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:06,993 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:06,995 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:06,997 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:07,003 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 16:52:07,016 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:07,019 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:07,020 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:07,020 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:07,020 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:07,024 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:07,508 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8942, 'RANGING': 13380, 'CONSOLIDATING': 8163, 'VOLATILE': 2253}  ambiguous=13 (total=32738) horizon=12
2026-05-01 16:52:07,511 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-05-01 16:52:07,641 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:07,644 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:07,645 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:07,645 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:07,646 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:07,648 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:08,107 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 7969, 'RANGING': 13888, 'CONSOLIDATING': 8178, 'VOLATILE': 2703}  ambiguous=13 (total=32738) horizon=12
2026-05-01 16:52:08,110 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-05-01 16:52:08,240 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:08,243 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:08,244 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:08,244 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:08,245 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:08,247 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:08,697 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8757, 'RANGING': 13514, 'CONSOLIDATING': 8167, 'VOLATILE': 2302}  ambiguous=13 (total=32740) horizon=12
2026-05-01 16:52:08,700 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-05-01 16:52:08,830 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:08,833 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:08,834 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:08,834 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:08,834 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:08,836 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:09,302 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8274, 'RANGING': 13643, 'CONSOLIDATING': 8178, 'VOLATILE': 2644}  ambiguous=13 (total=32739) horizon=12
2026-05-01 16:52:09,305 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-05-01 16:52:09,429 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:09,431 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:09,432 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:09,433 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:09,433 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:09,435 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:09,860 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8713, 'RANGING': 13564, 'CONSOLIDATING': 8168, 'VOLATILE': 2295}  ambiguous=13 (total=32740) horizon=12
2026-05-01 16:52:09,863 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-05-01 16:52:09,988 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:09,990 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:09,991 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:09,991 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:09,992 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:09,994 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:10,432 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8295, 'RANGING': 13768, 'CONSOLIDATING': 8178, 'VOLATILE': 2498}  ambiguous=13 (total=32739) horizon=12
2026-05-01 16:52:10,435 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-05-01 16:52:10,562 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-05-01 16:52:10,564 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-05-01 16:52:10,565 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-05-01 16:52:10,565 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-05-01 16:52:10,565 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-05-01 16:52:10,567 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:11,003 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 9045, 'RANGING': 13346, 'CONSOLIDATING': 8165, 'VOLATILE': 2183}  ambiguous=13 (total=32739) horizon=12
2026-05-01 16:52:11,007 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-05-01 16:52:11,140 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:11,142 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:11,143 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:11,143 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:11,144 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:11,146 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:11,578 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8133, 'RANGING': 13841, 'CONSOLIDATING': 8176, 'VOLATILE': 2590}  ambiguous=13 (total=32740) horizon=12
2026-05-01 16:52:11,582 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-05-01 16:52:11,712 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:11,714 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:11,715 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:11,715 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:11,716 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:11,718 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:12,167 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 7969, 'RANGING': 13883, 'CONSOLIDATING': 8177, 'VOLATILE': 2712}  ambiguous=13 (total=32741) horizon=12
2026-05-01 16:52:12,170 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-05-01 16:52:12,295 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:12,298 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:12,299 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:12,299 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:12,299 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-05-01 16:52:12,301 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-05-01 16:52:12,739 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 8927, 'RANGING': 13395, 'CONSOLIDATING': 8160, 'VOLATILE': 2261}  ambiguous=13 (total=32743) horizon=12
2026-05-01 16:52:12,742 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-05-01 16:52:12,881 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-05-01 16:52:12,885 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-05-01 16:52:12,886 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-05-01 16:52:12,887 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-05-01 16:52:12,887 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-05-01 16:52:12,891 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 16:52:13,826 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 19458, 'RANGING': 31056, 'CONSOLIDATING': 18647, 'VOLATILE': 5463}  ambiguous=13 (total=74624) horizon=12
2026-05-01 16:52:13,832 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-05-01 16:52:14,132 INFO Regime phase LTF dataset build: 7.2s (401471 samples)
2026-05-01 16:52:14,133 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/regime_ltf.pkl_20260501_165214
2026-05-01 16:52:14,140 INFO RegimeClassifier[mode=ltf_behaviour] loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl (device=cuda, features=18, n_classes=4)
2026-05-01 16:52:14,143 INFO RegimeClassifier[mode=ltf_behaviour]: dropped ambiguous labels below 0.40 (kept=401339 dropped=132 classes={'TRENDING': 104312, 'RANGING': 166920, 'CONSOLIDATING': 100239, 'VOLATILE': 29868})
2026-05-01 16:52:14,232 INFO RegimeClassifier[mode=ltf_behaviour]: 401339 samples, classes={'TRENDING': 104312, 'RANGING': 166920, 'CONSOLIDATING': 100239, 'VOLATILE': 29868}, device=cuda
2026-05-01 16:52:14,233 INFO RegimeClassifier: sample weights — mean=0.790  ambiguous(<0.4)=0.0%
2026-05-01 16:52:14,233 INFO RegimeClassifier[mode=ltf_behaviour]: warm start from existing weights
2026-05-01 16:52:14,233 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-05-01 16:52:14,956 INFO Regime epoch  1/50 — tr=0.6991 va=0.6334 acc=0.374 per_class={'TRENDING': 0.212, 'RANGING': 0.237, 'CONSOLIDATING': 0.691, 'VOLATILE': 0.641}
2026-05-01 16:52:15,625 INFO Regime epoch  2/50 — tr=0.6977 va=0.6351 acc=0.376
2026-05-01 16:52:16,274 INFO Regime epoch  3/50 — tr=0.6947 va=0.6354 acc=0.375
2026-05-01 16:52:16,918 INFO Regime epoch  4/50 — tr=0.6913 va=0.6364 acc=0.373
2026-05-01 16:52:17,599 INFO Regime epoch  5/50 — tr=0.6852 va=0.6382 acc=0.370 per_class={'TRENDING': 0.203, 'RANGING': 0.242, 'CONSOLIDATING': 0.671, 'VOLATILE': 0.657}
2026-05-01 16:52:18,245 INFO Regime epoch  6/50 — tr=0.6774 va=0.6381 acc=0.362
2026-05-01 16:52:18,866 INFO Regime epoch  7/50 — tr=0.6691 va=0.6391 acc=0.355
2026-05-01 16:52:19,507 INFO Regime epoch  8/50 — tr=0.6602 va=0.6398 acc=0.346
2026-05-01 16:52:20,158 INFO Regime epoch  9/50 — tr=0.6542 va=0.6414 acc=0.342
2026-05-01 16:52:20,872 INFO Regime epoch 10/50 — tr=0.6472 va=0.6420 acc=0.336 per_class={'TRENDING': 0.185, 'RANGING': 0.173, 'CONSOLIDATING': 0.645, 'VOLATILE': 0.739}
2026-05-01 16:52:21,521 INFO Regime epoch 11/50 — tr=0.6439 va=0.6418 acc=0.324
2026-05-01 16:52:21,521 INFO Regime early stop at epoch 11 (no_improve=10)
2026-05-01 16:52:21,570 WARNING RegimeClassifier accuracy 0.374 < warning floor 0.400 (harder structural labels; check blind backtest economics)
2026-05-01 16:52:21,574 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-01 16:52:21,574 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-01 16:52:21,577 INFO Regime phase LTF train: 7.4s
2026-05-01 16:52:21,701 INFO Regime LTF complete: acc=0.374, n=401471 per_class={'TRENDING': 0.212, 'RANGING': 0.237, 'CONSOLIDATING': 0.691, 'VOLATILE': 0.641}
2026-05-01 16:52:21,704 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-05-01 16:52:21,790 INFO Structural labels LTF_BEHAVIOUR [1H]: {'TRENDING': 19458, 'RANGING': 31056, 'CONSOLIDATING': 18647, 'VOLATILE': 5463}  ambiguous=13 (total=74624) horizon=12
2026-05-01 16:52:21,802 INFO Regime[1H mode=ltf_behaviour] persistence (avg bars/run) on XAUUSD 1H:
{'TRENDING': 4.125079499682001, 'RANGING': 4.718322698268004, 'CONSOLIDATING': 5.990041760359782, 'VOLATILE': 3.723926380368098}
2026-05-01 16:52:21,810 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (all labels):
{'TRENDING': {'n': 19458, 'mean': 5.273978151244844e-05, 'mean_over_std': 0.021533559889529843}, 'RANGING': {'n': 31055, 'mean': -2.3091416023246664e-05, 'mean_over_std': -0.012142968527728414}, 'CONSOLIDATING': {'n': 18647, 'mean': 6.38696766285663e-05, 'mean_over_std': 0.041981339266451534}, 'VOLATILE': {'n': 5463, 'mean': -0.00012433839470805868, 'mean_over_std': -0.04867063159870624}}
2026-05-01 16:52:21,811 INFO Regime[1H mode=ltf_behaviour] return separation on XAUUSD 1H (clean labels conf>=0.40):
{'TRENDING': {'n': 19458, 'mean': 5.273978151244844e-05, 'mean_over_std': 0.021533559889529843}, 'RANGING': {'n': 31043, 'mean': -2.2961083650504332e-05, 'mean_over_std': -0.01207348194887135}, 'CONSOLIDATING': {'n': 18647, 'mean': 6.38696766285663e-05, 'mean_over_std': 0.041981339266451534}, 'VOLATILE': {'n': 5463, 'mean': -0.00012433839470805868, 'mean_over_std': -0.04867063159870624}}
2026-05-01 16:52:21,820 INFO Regime retrain total: 36.8s (504761 samples)
2026-05-01 16:52:21,829 INFO Retrain complete. Total wall-clock: 36.8s

=== TRAINING COMPLETE ===
  regime: FAILED: exit 1
2026-05-01 16:52:24,167 ERROR retrain regime failed (exit 1)
2026-05-01 16:52:24,167 ERROR Model regime failed: exit 1
2026-05-01 16:52:24,168 INFO   [OK] gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt
2026-05-01 16:52:24,168 INFO   [OK] regime_htf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-05-01 16:52:24,168 INFO   [OK] regime_ltf → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-05-01 16:52:24,168 INFO   [OK] quality_scorer → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/quality_scorer.pkl
2026-05-01 16:52:24,168 INFO   [OK] rl_ppo → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/rl_ppo/model.zip
2026-05-01 16:52:24,168 INFO All Step 7a weights present in canonical location: /kaggle/working/Multi-Bot/trading-system/trading-engine/weights
2026-05-01 16:52:24,169 INFO Saved 17 retrain records to metrics/
2026-05-01 16:52:24,170 ERROR Step 7a failed; required training/artifacts missing: ['regime']
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