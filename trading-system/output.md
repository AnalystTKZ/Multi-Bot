All scripts and inputs verified.

=== Phase 0-5: Data preparation ===
  SKIP  Step 0 - Resample
  SKIP  Step 1 - Inventory
  SKIP  Step 2 - Cleaning
  SKIP  Step 3 - Alignment
  SKIP  Step 4 - Features
  START Step 5 - Split
2026-04-19 15:10:40,561 INFO Loading feature-engineered data...
2026-04-19 15:10:41,091 INFO Loaded 221743 rows, 202 features
2026-04-19 15:10:41,091 INFO Data span: 2016-01-04 → 2025-08-05  (9.6 years)
2026-04-19 15:10:41,092 INFO Train:        130951 bars  2016-01-04 → 2021-08-05
2026-04-19 15:10:41,092 INFO Validation:    44000 bars  2021-08-05 → 2023-08-04
2026-04-19 15:10:41,092 INFO Test:          46792 bars  2023-08-07 → 2025-08-05
2026-04-19 15:10:41,092 INFO No leakage confirmed: train < val < test timestamps

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
2026-04-19 15:10:43,396 INFO === STEP 7a: GRU + REGIME TRAINING ===
2026-04-19 15:10:43,396 INFO --- Training regime ---
2026-04-19 15:10:43,397 INFO Running retrain --model regime
2026-04-19 15:10:43,701 INFO retrain environment: KAGGLE
2026-04-19 15:10:45,396 INFO Device: CUDA (2 GPU(s))
2026-04-19 15:10:45,408 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 15:10:45,408 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 15:10:45,409 INFO === RegimeClassifier retrain (hierarchical: HTF 3-class bias + LTF 4-class behaviour) ===
2026-04-19 15:10:45,575 INFO NumExpr defaulting to 4 threads.
2026-04-19 15:10:45,782 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-19 15:10:45,782 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 15:10:45,782 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 15:10:45,984 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-19 15:10:45,986 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,062 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,133 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,203 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,275 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,349 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,416 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,490 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,560 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,632 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,721 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:10:46,781 INFO Regime: fitting per-group GMMs for HTF (dollar / cross / gold)...
2026-04-19 15:10:46,797 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,798 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,813 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,815 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,830 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,832 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,847 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,849 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,865 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,868 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,883 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,886 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,900 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,902 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,916 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,919 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,934 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,937 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,952 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,955 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:10:46,972 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:10:46,979 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:10:47,730 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 15:11:10,020 INFO GMM fitted on 58459 samples (mode=htf_bias) — cluster→regime: {2: 0, 0: 1, 1: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 15:11:10,024 INFO Regime HTF GMM 'dollar' fitted on 7 4H dfs (3-class bias)
2026-04-19 15:11:10,024 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 15:11:19,927 INFO GMM fitted on 25054 samples (mode=htf_bias) — cluster→regime: {1: 0, 0: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 15:11:19,931 INFO Regime HTF GMM 'cross' fitted on 3 4H dfs (3-class bias)
2026-04-19 15:11:19,931 INFO GMM fit: timeframe=4H mode=htf_bias → n_bar=50 n_components=3
2026-04-19 15:11:27,518 INFO GMM fitted on 19766 samples (mode=htf_bias) — cluster→regime: {0: 0, 1: 1, 2: 2} dist: {'BIAS_UP': 1, 'BIAS_DOWN': 1, 'BIAS_NEUTRAL': 1}
2026-04-19 15:11:27,524 INFO Regime HTF GMM 'gold' fitted on 1 4H dfs (3-class bias)
2026-04-19 15:11:27,524 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 15:12:37,461 INFO GMM fitted on 76337 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 3: 0, 0: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 15:12:37,468 INFO Regime LTF GMM 'dollar' fitted on 7 1H dfs (4-class behaviour)
2026-04-19 15:12:37,468 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 15:13:08,659 INFO GMM fitted on 32715 samples (mode=ltf_behaviour) — cluster→regime: {0: 3, 3: 0, 2: 2, 1: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 15:13:08,663 INFO Regime LTF GMM 'cross' fitted on 3 1H dfs (4-class behaviour)
2026-04-19 15:13:08,663 INFO GMM fit: timeframe=1H mode=ltf_behaviour → n_bar=24 n_components=4
2026-04-19 15:13:30,338 INFO GMM fitted on 10657 samples (mode=ltf_behaviour) — cluster→regime: {2: 3, 1: 0, 0: 2, 3: 1} dist: {'TRENDING': 1, 'RANGING': 1, 'CONSOLIDATING': 1, 'VOLATILE': 1}
2026-04-19 15:13:30,339 INFO Regime LTF GMM 'gold' fitted on 1 1H dfs (4-class behaviour)
2026-04-19 15:13:30,441 INFO Regime: training HTF bias classifier (3-class: BIAS_UP/DOWN/NEUTRAL)...
2026-04-19 15:13:30,443 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:30,444 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:30,445 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:30,446 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:30,447 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:30,448 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:30,449 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:30,450 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:30,451 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:30,452 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:30,453 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:13:30,578 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:30,621 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:30,622 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:30,623 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:30,632 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:30,633 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:31,058 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2439, 'BIAS_DOWN': 2349, 'BIAS_NEUTRAL': 3614}  ambiguous=4099 (total=8402)  short_runs_zeroed=649
2026-04-19 15:13:31,059 INFO Regime[4H mode=htf_bias]: collected AUDUSD — 8352 samples (group=dollar)
2026-04-19 15:13:31,237 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:31,270 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:31,271 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:31,272 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:31,282 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:31,283 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:31,637 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2503, 'BIAS_DOWN': 2397, 'BIAS_NEUTRAL': 3502}  ambiguous=4050 (total=8402)  short_runs_zeroed=744
2026-04-19 15:13:31,638 INFO Regime[4H mode=htf_bias]: collected EURGBP — 8352 samples (group=cross)
2026-04-19 15:13:31,821 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:31,857 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:31,858 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:31,858 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:31,867 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:31,868 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:32,227 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2482, 'BIAS_DOWN': 2360, 'BIAS_NEUTRAL': 3560}  ambiguous=3926 (total=8402)  short_runs_zeroed=522
2026-04-19 15:13:32,228 INFO Regime[4H mode=htf_bias]: collected EURJPY — 8352 samples (group=cross)
2026-04-19 15:13:32,397 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:32,434 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:32,434 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:32,435 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:32,444 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:32,445 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:32,795 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2460, 'BIAS_DOWN': 2376, 'BIAS_NEUTRAL': 3566}  ambiguous=3995 (total=8402)  short_runs_zeroed=585
2026-04-19 15:13:32,796 INFO Regime[4H mode=htf_bias]: collected EURUSD — 8352 samples (group=dollar)
2026-04-19 15:13:32,984 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:33,020 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:33,021 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:33,022 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:33,030 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:33,031 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:33,382 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2348, 'BIAS_DOWN': 2535, 'BIAS_NEUTRAL': 3520}  ambiguous=3831 (total=8403)  short_runs_zeroed=439
2026-04-19 15:13:33,383 INFO Regime[4H mode=htf_bias]: collected GBPJPY — 8353 samples (group=cross)
2026-04-19 15:13:33,557 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:33,607 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:33,608 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:33,608 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:33,617 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:33,619 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:33,975 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2440, 'BIAS_DOWN': 2280, 'BIAS_NEUTRAL': 3683}  ambiguous=4047 (total=8403)  short_runs_zeroed=508
2026-04-19 15:13:33,977 INFO Regime[4H mode=htf_bias]: collected GBPUSD — 8353 samples (group=dollar)
2026-04-19 15:13:34,145 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:13:34,174 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:13:34,174 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:13:34,175 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:13:34,183 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:13:34,184 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:34,540 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2526, 'BIAS_DOWN': 2331, 'BIAS_NEUTRAL': 3545}  ambiguous=3965 (total=8402)  short_runs_zeroed=561
2026-04-19 15:13:34,541 INFO Regime[4H mode=htf_bias]: collected NZDUSD — 8352 samples (group=dollar)
2026-04-19 15:13:34,723 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:34,758 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:34,759 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:34,759 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:34,768 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:34,769 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:35,118 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2250, 'BIAS_DOWN': 2599, 'BIAS_NEUTRAL': 3553}  ambiguous=3919 (total=8402)  short_runs_zeroed=547
2026-04-19 15:13:35,119 INFO Regime[4H mode=htf_bias]: collected USDCAD — 8352 samples (group=dollar)
2026-04-19 15:13:35,282 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:35,316 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:35,317 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:35,317 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:35,327 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:35,328 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:35,665 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2214, 'BIAS_DOWN': 2634, 'BIAS_NEUTRAL': 3554}  ambiguous=3934 (total=8402)  short_runs_zeroed=549
2026-04-19 15:13:35,666 INFO Regime[4H mode=htf_bias]: collected USDCHF — 8352 samples (group=dollar)
2026-04-19 15:13:35,837 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:35,874 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:35,874 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:35,875 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:35,884 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:35,885 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:36,234 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2278, 'BIAS_DOWN': 2515, 'BIAS_NEUTRAL': 3610}  ambiguous=3986 (total=8403)  short_runs_zeroed=516
2026-04-19 15:13:36,235 INFO Regime[4H mode=htf_bias]: collected USDJPY — 8353 samples (group=dollar)
2026-04-19 15:13:36,518 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:13:36,582 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:13:36,583 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:13:36,584 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:13:36,596 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:13:36,598 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:13:37,360 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 6629, 'BIAS_DOWN': 5241, 'BIAS_NEUTRAL': 7947}  ambiguous=8645 (total=19817)  short_runs_zeroed=980
2026-04-19 15:13:37,362 INFO Regime[4H mode=htf_bias]: collected XAUUSD — 19767 samples (group=gold)
2026-04-19 15:13:37,538 INFO RegimeClassifier[mode=htf_bias]: 103290 samples, classes={'BIAS_UP': 30569, 'BIAS_DOWN': 29617, 'BIAS_NEUTRAL': 43104}, device=cuda
2026-04-19 15:13:37,539 INFO RegimeClassifier: sample weights — mean=0.413  ambiguous(<0.4)=47.0%
2026-04-19 15:13:37,732 INFO RegimeClassifier[mode=htf_bias]: cold start (no existing weights)
2026-04-19 15:13:37,732 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-19 15:13:40,148 INFO Regime epoch  1/50 — tr=0.5852 va=1.0692 acc=0.493 per_class={'BIAS_UP': 0.619, 'BIAS_DOWN': 0.386, 'BIAS_NEUTRAL': 0.454}
2026-04-19 15:13:40,318 INFO Regime epoch  2/50 — tr=0.5748 va=1.0367 acc=0.547
2026-04-19 15:13:40,493 INFO Regime epoch  3/50 — tr=0.5554 va=0.9845 acc=0.625
2026-04-19 15:13:40,664 INFO Regime epoch  4/50 — tr=0.5277 va=0.9141 acc=0.664
2026-04-19 15:13:40,847 INFO Regime epoch  5/50 — tr=0.4961 va=0.8380 acc=0.679 per_class={'BIAS_UP': 0.97, 'BIAS_DOWN': 0.961, 'BIAS_NEUTRAL': 0.248}
2026-04-19 15:13:41,038 INFO Regime epoch  6/50 — tr=0.4668 va=0.7675 acc=0.697
2026-04-19 15:13:41,208 INFO Regime epoch  7/50 — tr=0.4462 va=0.7153 acc=0.710
2026-04-19 15:13:41,377 INFO Regime epoch  8/50 — tr=0.4329 va=0.6854 acc=0.715
2026-04-19 15:13:41,544 INFO Regime epoch  9/50 — tr=0.4257 va=0.6744 acc=0.706
2026-04-19 15:13:41,724 INFO Regime epoch 10/50 — tr=0.4199 va=0.6643 acc=0.715 per_class={'BIAS_UP': 0.998, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.289}
2026-04-19 15:13:41,889 INFO Regime epoch 11/50 — tr=0.4156 va=0.6610 acc=0.717
2026-04-19 15:13:42,060 INFO Regime epoch 12/50 — tr=0.4122 va=0.6625 acc=0.711
2026-04-19 15:13:42,236 INFO Regime epoch 13/50 — tr=0.4095 va=0.6623 acc=0.713
2026-04-19 15:13:42,408 INFO Regime epoch 14/50 — tr=0.4076 va=0.6608 acc=0.712
2026-04-19 15:13:42,594 INFO Regime epoch 15/50 — tr=0.4057 va=0.6608 acc=0.714 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.284}
2026-04-19 15:13:42,771 INFO Regime epoch 16/50 — tr=0.4041 va=0.6591 acc=0.711
2026-04-19 15:13:42,946 INFO Regime epoch 17/50 — tr=0.4023 va=0.6586 acc=0.715
2026-04-19 15:13:43,130 INFO Regime epoch 18/50 — tr=0.4012 va=0.6582 acc=0.714
2026-04-19 15:13:43,299 INFO Regime epoch 19/50 — tr=0.4004 va=0.6575 acc=0.716
2026-04-19 15:13:43,485 INFO Regime epoch 20/50 — tr=0.3991 va=0.6546 acc=0.719 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.297}
2026-04-19 15:13:43,670 INFO Regime epoch 21/50 — tr=0.3984 va=0.6543 acc=0.718
2026-04-19 15:13:43,847 INFO Regime epoch 22/50 — tr=0.3975 va=0.6540 acc=0.719
2026-04-19 15:13:44,025 INFO Regime epoch 23/50 — tr=0.3969 va=0.6536 acc=0.719
2026-04-19 15:13:44,197 INFO Regime epoch 24/50 — tr=0.3963 va=0.6526 acc=0.719
2026-04-19 15:13:44,384 INFO Regime epoch 25/50 — tr=0.3960 va=0.6529 acc=0.719 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.297}
2026-04-19 15:13:44,556 INFO Regime epoch 26/50 — tr=0.3957 va=0.6520 acc=0.720
2026-04-19 15:13:44,734 INFO Regime epoch 27/50 — tr=0.3951 va=0.6528 acc=0.724
2026-04-19 15:13:44,919 INFO Regime epoch 28/50 — tr=0.3948 va=0.6510 acc=0.723
2026-04-19 15:13:45,104 INFO Regime epoch 29/50 — tr=0.3945 va=0.6498 acc=0.725
2026-04-19 15:13:45,299 INFO Regime epoch 30/50 — tr=0.3941 va=0.6488 acc=0.723 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.307}
2026-04-19 15:13:45,469 INFO Regime epoch 31/50 — tr=0.3939 va=0.6503 acc=0.725
2026-04-19 15:13:45,645 INFO Regime epoch 32/50 — tr=0.3936 va=0.6497 acc=0.727
2026-04-19 15:13:45,817 INFO Regime epoch 33/50 — tr=0.3934 va=0.6496 acc=0.725
2026-04-19 15:13:45,988 INFO Regime epoch 34/50 — tr=0.3932 va=0.6481 acc=0.727
2026-04-19 15:13:46,181 INFO Regime epoch 35/50 — tr=0.3929 va=0.6480 acc=0.726 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.315}
2026-04-19 15:13:46,368 INFO Regime epoch 36/50 — tr=0.3929 va=0.6491 acc=0.727
2026-04-19 15:13:46,543 INFO Regime epoch 37/50 — tr=0.3926 va=0.6481 acc=0.725
2026-04-19 15:13:46,709 INFO Regime epoch 38/50 — tr=0.3925 va=0.6459 acc=0.726
2026-04-19 15:13:46,886 INFO Regime epoch 39/50 — tr=0.3926 va=0.6490 acc=0.726
2026-04-19 15:13:47,068 INFO Regime epoch 40/50 — tr=0.3924 va=0.6473 acc=0.729 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.322}
2026-04-19 15:13:47,235 INFO Regime epoch 41/50 — tr=0.3924 va=0.6460 acc=0.728
2026-04-19 15:13:47,411 INFO Regime epoch 42/50 — tr=0.3923 va=0.6472 acc=0.728
2026-04-19 15:13:47,586 INFO Regime epoch 43/50 — tr=0.3924 va=0.6482 acc=0.730
2026-04-19 15:13:47,764 INFO Regime epoch 44/50 — tr=0.3924 va=0.6487 acc=0.726
2026-04-19 15:13:47,952 INFO Regime epoch 45/50 — tr=0.3922 va=0.6478 acc=0.730 per_class={'BIAS_UP': 1.0, 'BIAS_DOWN': 1.0, 'BIAS_NEUTRAL': 0.325}
2026-04-19 15:13:48,130 INFO Regime epoch 46/50 — tr=0.3922 va=0.6461 acc=0.730
2026-04-19 15:13:48,309 INFO Regime epoch 47/50 — tr=0.3921 va=0.6477 acc=0.729
2026-04-19 15:13:48,488 INFO Regime epoch 48/50 — tr=0.3920 va=0.6501 acc=0.727
2026-04-19 15:13:48,488 INFO Regime early stop at epoch 48 (no_improve=10)
2026-04-19 15:13:48,503 INFO RegimeClassifier[mode=htf_bias] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-19 15:13:48,503 INFO RegimeClassifier[4H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_htf.pkl
2026-04-19 15:13:48,628 INFO Regime HTF complete: acc=0.726, n=103290
2026-04-19 15:13:48,630 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:13:48,781 INFO Rule labels LTF_BEHAVIOUR [4H]: {'TRENDING': 8970, 'RANGING': 4592, 'CONSOLIDATING': 2247, 'VOLATILE': 4008}  ambiguous=4132 (total=19817)  short_runs_zeroed=570
2026-04-19 15:13:48,783 INFO Regime[4H] persistence (avg bars/run) on XAUUSD 4H:
{0: 12.121621621621621, 1: 6.4858757062146895, 2: 5.575682382133995, 3: 10.329896907216495}
2026-04-19 15:13:48,785 INFO Regime[4H] return separation on XAUUSD 4H:
{0: 5.9502411776840314e-05, 1: 5.664222849552333e-05, 2: -6.138442068010657e-05, 3: 4.239843820627722e-05}
2026-04-19 15:13:48,785 INFO Regime: training LTF behaviour classifier (4-class: TRENDING/RANGING/CONSOLIDATING/VOLATILE)...
2026-04-19 15:13:48,787 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:48,789 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:48,790 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:48,792 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:48,794 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:48,795 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:48,796 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:48,798 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:48,800 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:48,801 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:48,804 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:13:48,814 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:48,816 INFO Loaded AUDUSD/15M split=all: 234948 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:48,817 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:48,817 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:48,817 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:48,820 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:49,394 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=10885 (total=32738)  short_runs_zeroed=4986
2026-04-19 15:13:49,397 INFO Regime[1H mode=ltf_behaviour]: collected AUDUSD — 32688 samples (group=dollar)
2026-04-19 15:13:49,529 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:49,532 INFO Loaded EURGBP/15M split=all: 234979 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:49,532 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:49,533 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:49,533 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:49,535 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:50,081 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=10483 (total=32738)  short_runs_zeroed=4347
2026-04-19 15:13:50,084 INFO Regime[1H mode=ltf_behaviour]: collected EURGBP — 32688 samples (group=cross)
2026-04-19 15:13:50,222 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:50,225 INFO Loaded EURJPY/15M split=all: 234916 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:50,225 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:50,226 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:50,226 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:50,228 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:50,774 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10628 (total=32740)  short_runs_zeroed=4399
2026-04-19 15:13:50,777 INFO Regime[1H mode=ltf_behaviour]: collected EURJPY — 32690 samples (group=cross)
2026-04-19 15:13:50,916 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:50,918 INFO Loaded EURUSD/15M split=all: 235026 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:50,919 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:50,920 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:50,920 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:50,922 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:51,485 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10080 (total=32739)  short_runs_zeroed=3955
2026-04-19 15:13:51,487 INFO Regime[1H mode=ltf_behaviour]: collected EURUSD — 32689 samples (group=dollar)
2026-04-19 15:13:51,617 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:51,621 INFO Loaded GBPJPY/15M split=all: 234918 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:51,622 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:51,622 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:51,623 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:51,624 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:52,172 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10370 (total=32740)  short_runs_zeroed=4397
2026-04-19 15:13:52,175 INFO Regime[1H mode=ltf_behaviour]: collected GBPJPY — 32690 samples (group=cross)
2026-04-19 15:13:52,308 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:52,310 INFO Loaded GBPUSD/15M split=all: 234968 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:52,311 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:52,312 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:52,312 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:52,314 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:52,884 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=9824 (total=32739)  short_runs_zeroed=3724
2026-04-19 15:13:52,887 INFO Regime[1H mode=ltf_behaviour]: collected GBPUSD — 32689 samples (group=dollar)
2026-04-19 15:13:53,026 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:13:53,027 INFO Loaded NZDUSD/15M split=all: 174689 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:13:53,028 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:13:53,028 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:13:53,029 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:13:53,030 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:53,570 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10686 (total=32739)  short_runs_zeroed=4898
2026-04-19 15:13:53,573 INFO Regime[1H mode=ltf_behaviour]: collected NZDUSD — 32689 samples (group=dollar)
2026-04-19 15:13:53,704 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:53,707 INFO Loaded USDCAD/15M split=all: 234962 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:53,707 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:53,708 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:53,708 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:53,710 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:54,256 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=9919 (total=32740)  short_runs_zeroed=3880
2026-04-19 15:13:54,259 INFO Regime[1H mode=ltf_behaviour]: collected USDCAD — 32690 samples (group=dollar)
2026-04-19 15:13:54,398 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:54,400 INFO Loaded USDCHF/15M split=all: 234958 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:54,401 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:54,402 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:54,402 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:54,404 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:54,953 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=10596 (total=32741)  short_runs_zeroed=3896
2026-04-19 15:13:54,956 INFO Regime[1H mode=ltf_behaviour]: collected USDCHF — 32691 samples (group=dollar)
2026-04-19 15:13:55,091 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:55,095 INFO Loaded USDJPY/15M split=all: 234955 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:55,096 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:55,096 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:55,097 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:13:55,099 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:13:55,652 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10093 (total=32743)  short_runs_zeroed=4275
2026-04-19 15:13:55,654 INFO Regime[1H mode=ltf_behaviour]: collected USDJPY — 32693 samples (group=dollar)
2026-04-19 15:13:55,797 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:13:55,804 INFO Loaded XAUUSD/15M split=all: 401431 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:13:55,805 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:13:55,806 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:13:55,806 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:13:55,809 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:13:57,007 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=23184 (total=74624)  short_runs_zeroed=9134
2026-04-19 15:13:57,012 INFO Regime[1H mode=ltf_behaviour]: collected XAUUSD — 74574 samples (group=gold)
2026-04-19 15:13:57,374 INFO RegimeClassifier[mode=ltf_behaviour]: 401471 samples, classes={'TRENDING': 160094, 'RANGING': 105286, 'CONSOLIDATING': 53524, 'VOLATILE': 82567}, device=cuda
2026-04-19 15:13:57,375 INFO RegimeClassifier: sample weights — mean=0.532  ambiguous(<0.4)=31.6%
2026-04-19 15:13:57,377 INFO RegimeClassifier[mode=ltf_behaviour]: cold start (no existing weights)
2026-04-19 15:13:57,377 INFO RegimeClassifier: DataParallel across 2 GPUs
2026-04-19 15:13:58,071 INFO Regime epoch  1/50 — tr=0.7970 va=1.3370 acc=0.339 per_class={'TRENDING': 0.321, 'RANGING': 0.327, 'CONSOLIDATING': 0.576, 'VOLATILE': 0.228}
2026-04-19 15:13:58,697 INFO Regime epoch  2/50 — tr=0.7654 va=1.2460 acc=0.435
2026-04-19 15:13:59,356 INFO Regime epoch  3/50 — tr=0.7223 va=1.1614 acc=0.438
2026-04-19 15:14:00,003 INFO Regime epoch  4/50 — tr=0.6845 va=1.1080 acc=0.445
2026-04-19 15:14:00,710 INFO Regime epoch  5/50 — tr=0.6574 va=1.0787 acc=0.458 per_class={'TRENDING': 0.388, 'RANGING': 0.024, 'CONSOLIDATING': 0.834, 'VOLATILE': 0.911}
2026-04-19 15:14:01,402 INFO Regime epoch  6/50 — tr=0.6410 va=1.0610 acc=0.481
2026-04-19 15:14:02,061 INFO Regime epoch  7/50 — tr=0.6295 va=1.0477 acc=0.498
2026-04-19 15:14:02,720 INFO Regime epoch  8/50 — tr=0.6206 va=1.0349 acc=0.514
2026-04-19 15:14:03,357 INFO Regime epoch  9/50 — tr=0.6134 va=1.0203 acc=0.523
2026-04-19 15:14:04,058 INFO Regime epoch 10/50 — tr=0.6077 va=1.0083 acc=0.530 per_class={'TRENDING': 0.566, 'RANGING': 0.013, 'CONSOLIDATING': 0.866, 'VOLATILE': 0.909}
2026-04-19 15:14:04,726 INFO Regime epoch 11/50 — tr=0.6033 va=0.9957 acc=0.540
2026-04-19 15:14:05,360 INFO Regime epoch 12/50 — tr=0.5988 va=0.9854 acc=0.547
2026-04-19 15:14:05,989 INFO Regime epoch 13/50 — tr=0.5954 va=0.9781 acc=0.550
2026-04-19 15:14:06,614 INFO Regime epoch 14/50 — tr=0.5919 va=0.9671 acc=0.557
2026-04-19 15:14:07,318 INFO Regime epoch 15/50 — tr=0.5897 va=0.9614 acc=0.565 per_class={'TRENDING': 0.642, 'RANGING': 0.003, 'CONSOLIDATING': 0.918, 'VOLATILE': 0.909}
2026-04-19 15:14:07,959 INFO Regime epoch 16/50 — tr=0.5871 va=0.9578 acc=0.566
2026-04-19 15:14:08,602 INFO Regime epoch 17/50 — tr=0.5849 va=0.9507 acc=0.571
2026-04-19 15:14:09,253 INFO Regime epoch 18/50 — tr=0.5827 va=0.9500 acc=0.570
2026-04-19 15:14:09,923 INFO Regime epoch 19/50 — tr=0.5813 va=0.9451 acc=0.573
2026-04-19 15:14:10,636 INFO Regime epoch 20/50 — tr=0.5799 va=0.9444 acc=0.574 per_class={'TRENDING': 0.649, 'RANGING': 0.001, 'CONSOLIDATING': 0.947, 'VOLATILE': 0.923}
2026-04-19 15:14:11,326 INFO Regime epoch 21/50 — tr=0.5785 va=0.9365 acc=0.579
2026-04-19 15:14:11,978 INFO Regime epoch 22/50 — tr=0.5776 va=0.9360 acc=0.580
2026-04-19 15:14:12,616 INFO Regime epoch 23/50 — tr=0.5765 va=0.9366 acc=0.576
2026-04-19 15:14:13,261 INFO Regime epoch 24/50 — tr=0.5758 va=0.9304 acc=0.585
2026-04-19 15:14:13,949 INFO Regime epoch 25/50 — tr=0.5752 va=0.9315 acc=0.582 per_class={'TRENDING': 0.666, 'RANGING': 0.0, 'CONSOLIDATING': 0.96, 'VOLATILE': 0.921}
2026-04-19 15:14:14,600 INFO Regime epoch 26/50 — tr=0.5744 va=0.9284 acc=0.582
2026-04-19 15:14:15,251 INFO Regime epoch 27/50 — tr=0.5739 va=0.9279 acc=0.582
2026-04-19 15:14:15,888 INFO Regime epoch 28/50 — tr=0.5733 va=0.9304 acc=0.579
2026-04-19 15:14:16,554 INFO Regime epoch 29/50 — tr=0.5729 va=0.9256 acc=0.581
2026-04-19 15:14:17,264 INFO Regime epoch 30/50 — tr=0.5725 va=0.9257 acc=0.584 per_class={'TRENDING': 0.67, 'RANGING': 0.0, 'CONSOLIDATING': 0.962, 'VOLATILE': 0.921}
2026-04-19 15:14:17,893 INFO Regime epoch 31/50 — tr=0.5727 va=0.9240 acc=0.586
2026-04-19 15:14:18,541 INFO Regime epoch 32/50 — tr=0.5721 va=0.9281 acc=0.581
2026-04-19 15:14:19,196 INFO Regime epoch 33/50 — tr=0.5718 va=0.9215 acc=0.587
2026-04-19 15:14:19,839 INFO Regime epoch 34/50 — tr=0.5716 va=0.9251 acc=0.582
2026-04-19 15:14:20,541 INFO Regime epoch 35/50 — tr=0.5714 va=0.9258 acc=0.583 per_class={'TRENDING': 0.665, 'RANGING': 0.001, 'CONSOLIDATING': 0.959, 'VOLATILE': 0.926}
2026-04-19 15:14:21,244 INFO Regime epoch 36/50 — tr=0.5711 va=0.9244 acc=0.586
2026-04-19 15:14:21,918 INFO Regime epoch 37/50 — tr=0.5711 va=0.9210 acc=0.589
2026-04-19 15:14:22,559 INFO Regime epoch 38/50 — tr=0.5709 va=0.9253 acc=0.583
2026-04-19 15:14:23,207 INFO Regime epoch 39/50 — tr=0.5709 va=0.9198 acc=0.586
2026-04-19 15:14:23,908 INFO Regime epoch 40/50 — tr=0.5708 va=0.9201 acc=0.587 per_class={'TRENDING': 0.679, 'RANGING': 0.001, 'CONSOLIDATING': 0.966, 'VOLATILE': 0.917}
2026-04-19 15:14:24,574 INFO Regime epoch 41/50 — tr=0.5707 va=0.9189 acc=0.586
2026-04-19 15:14:25,251 INFO Regime epoch 42/50 — tr=0.5706 va=0.9174 acc=0.588
2026-04-19 15:14:25,917 INFO Regime epoch 43/50 — tr=0.5707 va=0.9239 acc=0.585
2026-04-19 15:14:26,596 INFO Regime epoch 44/50 — tr=0.5706 va=0.9207 acc=0.586
2026-04-19 15:14:27,301 INFO Regime epoch 45/50 — tr=0.5706 va=0.9187 acc=0.586 per_class={'TRENDING': 0.673, 'RANGING': 0.001, 'CONSOLIDATING': 0.965, 'VOLATILE': 0.921}
2026-04-19 15:14:27,955 INFO Regime epoch 46/50 — tr=0.5705 va=0.9232 acc=0.583
2026-04-19 15:14:28,646 INFO Regime epoch 47/50 — tr=0.5704 va=0.9191 acc=0.586
2026-04-19 15:14:29,316 INFO Regime epoch 48/50 — tr=0.5703 va=0.9196 acc=0.587
2026-04-19 15:14:29,961 INFO Regime epoch 49/50 — tr=0.5704 va=0.9211 acc=0.586
2026-04-19 15:14:30,684 INFO Regime epoch 50/50 — tr=0.5701 va=0.9223 acc=0.585 per_class={'TRENDING': 0.672, 'RANGING': 0.001, 'CONSOLIDATING': 0.963, 'VOLATILE': 0.922}
2026-04-19 15:14:30,727 WARNING RegimeClassifier accuracy 0.59 < 0.65 threshold
2026-04-19 15:14:30,730 INFO RegimeClassifier[mode=ltf_behaviour] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-19 15:14:30,730 INFO RegimeClassifier[1H] saved to /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/regime_ltf.pkl
2026-04-19 15:14:30,862 INFO Regime LTF complete: acc=0.588, n=401471
2026-04-19 15:14:30,866 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:14:31,383 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=23184 (total=74624)  short_runs_zeroed=9134
2026-04-19 15:14:31,387 INFO Regime[1H] persistence (avg bars/run) on XAUUSD 1H:
{0: 9.355291913830783, 1: 6.096303199751476, 2: 5.598885793871866, 3: 6.771351107094442}
2026-04-19 15:14:31,390 INFO Regime[1H] return separation on XAUUSD 1H:
{0: 2.149062259083323e-05, 1: -3.254493608872846e-06, 2: -3.039756158146958e-06, 3: 1.8102764458593193e-05}
2026-04-19 15:14:31,402 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-19 15:14:31,402 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 15:14:31,402 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 15:14:31,402 INFO === VectorStore: building similarity indices ===
2026-04-19 15:14:31,403 ERROR _index_embeddings_post_train failed (non-fatal): faiss not installed. Install with: pip install faiss-gpu  (or faiss-cpu)
2026-04-19 15:14:31,403 INFO Retrain complete.
2026-04-19 15:14:32,444 INFO Model regime: SUCCESS
2026-04-19 15:14:32,444 INFO --- Training gru ---
2026-04-19 15:14:32,444 INFO Running retrain --model gru
2026-04-19 15:14:32,803 INFO retrain environment: KAGGLE
2026-04-19 15:14:34,471 INFO Device: CUDA (2 GPU(s))
2026-04-19 15:14:34,482 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 15:14:34,482 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 15:14:34,483 INFO === GRU-LSTM retrain (timeframes: ['5M', '15M', '1H', '4H']) ===
2026-04-19 15:14:34,625 INFO NumExpr defaulting to 4 threads.
2026-04-19 15:14:34,824 INFO GRU: 2 CUDA device(s) available — using GPU
2026-04-19 15:14:34,824 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 15:14:34,824 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 15:14:35,061 INFO GRULSTMPredictor loaded from /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm/model.pt (device=cuda)
2026-04-19 15:14:35,286 INFO Split boundaries loaded — train≤2021-08-05  val≤2023-08-04  test≤2025-08-05
2026-04-19 15:14:35,288 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:35,366 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:35,441 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:35,516 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:35,592 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:35,666 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:35,737 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:35,810 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:35,883 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:35,960 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:36,049 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:14:36,114 INFO GRU multi-symbol training (Kaggle mode): 11 symbols × ['5M', '15M', '1H', '4H']
2026-04-19 15:14:36,116 INFO Backed up /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/gru_lstm → /kaggle/working/Multi-Bot/trading-system/trading-engine/weights/backups/gru_lstm_20260419_151436
2026-04-19 15:14:36,119 INFO GRU feature contract unchanged (input_size=74) — incremental retrain
2026-04-19 15:14:36,246 INFO Loaded AUDUSD/5M split=all: 704678 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:36,247 INFO Loaded AUDUSD/1H split=all: 58741 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:36,262 INFO Loaded AUDUSD/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:36,276 INFO Loaded AUDUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:36,277 INFO RegimeClassifier: 2 GPU(s) available — training on CUDA
2026-04-19 15:14:36,278 INFO   GPU 0: Tesla T4 (15.6 GB)
2026-04-19 15:14:36,278 INFO   GPU 1: Tesla T4 (15.6 GB)
2026-04-19 15:14:36,279 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:36,358 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2439, 'BIAS_DOWN': 2349, 'BIAS_NEUTRAL': 3614}  ambiguous=4099 (total=8402)  short_runs_zeroed=649
2026-04-19 15:14:36,360 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:36,608 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13046, 'RANGING': 8399, 'CONSOLIDATING': 4410, 'VOLATILE': 6883}  ambiguous=10885 (total=32738)  short_runs_zeroed=4986
2026-04-19 15:14:36,636 INFO Loaded AUDUSD/5M split=train: 392782 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:36,916 INFO Loaded AUDUSD/15M split=train: 130944 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:37,041 INFO Loaded AUDUSD/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:37,132 INFO Loaded AUDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:37,337 INFO Loaded EURGBP/5M split=all: 704756 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:37,338 INFO Loaded EURGBP/1H split=all: 58748 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:37,353 INFO Loaded EURGBP/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:37,361 INFO Loaded EURGBP/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:37,362 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:37,432 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2503, 'BIAS_DOWN': 2397, 'BIAS_NEUTRAL': 3502}  ambiguous=4050 (total=8402)  short_runs_zeroed=744
2026-04-19 15:14:37,434 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:37,664 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12945, 'RANGING': 9013, 'CONSOLIDATING': 4150, 'VOLATILE': 6630}  ambiguous=10483 (total=32738)  short_runs_zeroed=4347
2026-04-19 15:14:37,679 INFO Loaded EURGBP/5M split=train: 392761 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:37,933 INFO Loaded EURGBP/15M split=train: 130945 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:38,057 INFO Loaded EURGBP/1H split=train: 32738 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:38,152 INFO Loaded EURGBP/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:38,340 INFO Loaded EURJPY/5M split=all: 704417 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:38,341 INFO Loaded EURJPY/1H split=all: 58735 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:38,358 INFO Loaded EURJPY/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:38,366 INFO Loaded EURJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:38,367 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:38,439 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2482, 'BIAS_DOWN': 2360, 'BIAS_NEUTRAL': 3560}  ambiguous=3926 (total=8402)  short_runs_zeroed=522
2026-04-19 15:14:38,441 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:38,672 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12559, 'RANGING': 8639, 'CONSOLIDATING': 4711, 'VOLATILE': 6831}  ambiguous=10628 (total=32740)  short_runs_zeroed=4399
2026-04-19 15:14:38,688 INFO Loaded EURJPY/5M split=train: 392828 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:38,957 INFO Loaded EURJPY/15M split=train: 130956 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:39,088 INFO Loaded EURJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:39,186 INFO Loaded EURJPY/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:39,376 INFO Loaded EURUSD/5M split=all: 704977 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:39,376 INFO Loaded EURUSD/1H split=all: 58760 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:39,392 INFO Loaded EURUSD/4H split=all: 15258 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:39,400 INFO Loaded EURUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:39,401 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:39,472 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2460, 'BIAS_DOWN': 2376, 'BIAS_NEUTRAL': 3566}  ambiguous=3995 (total=8402)  short_runs_zeroed=585
2026-04-19 15:14:39,474 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:39,702 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13098, 'RANGING': 8495, 'CONSOLIDATING': 4264, 'VOLATILE': 6882}  ambiguous=10080 (total=32739)  short_runs_zeroed=3955
2026-04-19 15:14:39,725 INFO Loaded EURUSD/5M split=train: 392826 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:39,985 INFO Loaded EURUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:40,113 INFO Loaded EURUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:40,209 INFO Loaded EURUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:40,385 INFO Loaded GBPJPY/5M split=all: 704330 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:40,386 INFO Loaded GBPJPY/1H split=all: 58736 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:40,403 INFO Loaded GBPJPY/4H split=all: 15259 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:40,410 INFO Loaded GBPJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:40,411 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:40,482 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2348, 'BIAS_DOWN': 2535, 'BIAS_NEUTRAL': 3520}  ambiguous=3831 (total=8403)  short_runs_zeroed=439
2026-04-19 15:14:40,483 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:40,736 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13031, 'RANGING': 8429, 'CONSOLIDATING': 4358, 'VOLATILE': 6922}  ambiguous=10370 (total=32740)  short_runs_zeroed=4397
2026-04-19 15:14:40,751 INFO Loaded GBPJPY/5M split=train: 392739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:41,035 INFO Loaded GBPJPY/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:41,166 INFO Loaded GBPJPY/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:41,264 INFO Loaded GBPJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:41,445 INFO Loaded GBPUSD/5M split=all: 704770 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:41,445 INFO Loaded GBPUSD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:41,461 INFO Loaded GBPUSD/4H split=all: 15256 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:41,469 INFO Loaded GBPUSD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:41,470 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:41,546 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2440, 'BIAS_DOWN': 2280, 'BIAS_NEUTRAL': 3683}  ambiguous=4047 (total=8403)  short_runs_zeroed=508
2026-04-19 15:14:41,548 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:41,777 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13541, 'RANGING': 8392, 'CONSOLIDATING': 4176, 'VOLATILE': 6630}  ambiguous=9824 (total=32739)  short_runs_zeroed=3724
2026-04-19 15:14:41,792 INFO Loaded GBPUSD/5M split=train: 392811 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:42,062 INFO Loaded GBPUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:42,193 INFO Loaded GBPUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:42,290 INFO Loaded GBPUSD/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:42,452 INFO Loaded NZDUSD/5M split=all: 523942 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:14:42,453 INFO Loaded NZDUSD/1H split=all: 43675 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:14:42,467 INFO Loaded NZDUSD/4H split=all: 11210 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:14:42,474 INFO Loaded NZDUSD/1D split=all: 1965 bars (2016-01-04 → 2025-08-05)
2026-04-19 15:14:42,475 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:42,550 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2526, 'BIAS_DOWN': 2331, 'BIAS_NEUTRAL': 3545}  ambiguous=3965 (total=8402)  short_runs_zeroed=561
2026-04-19 15:14:42,552 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:42,781 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13234, 'RANGING': 8307, 'CONSOLIDATING': 4424, 'VOLATILE': 6774}  ambiguous=10686 (total=32739)  short_runs_zeroed=4898
2026-04-19 15:14:42,793 INFO Loaded NZDUSD/5M split=train: 392773 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:43,051 INFO Loaded NZDUSD/15M split=train: 130951 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:43,176 INFO Loaded NZDUSD/1H split=train: 32739 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:43,274 INFO Loaded NZDUSD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:43,450 INFO Loaded USDCAD/5M split=all: 704701 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:43,451 INFO Loaded USDCAD/1H split=all: 58746 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:43,466 INFO Loaded USDCAD/4H split=all: 15255 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:43,474 INFO Loaded USDCAD/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:43,475 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:43,553 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2250, 'BIAS_DOWN': 2599, 'BIAS_NEUTRAL': 3553}  ambiguous=3919 (total=8402)  short_runs_zeroed=547
2026-04-19 15:14:43,555 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:43,787 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13326, 'RANGING': 8606, 'CONSOLIDATING': 3935, 'VOLATILE': 6873}  ambiguous=9919 (total=32740)  short_runs_zeroed=3880
2026-04-19 15:14:43,804 INFO Loaded USDCAD/5M split=train: 392802 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:44,069 INFO Loaded USDCAD/15M split=train: 130953 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:44,195 INFO Loaded USDCAD/1H split=train: 32740 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:44,286 INFO Loaded USDCAD/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:44,462 INFO Loaded USDCHF/5M split=all: 704572 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:44,462 INFO Loaded USDCHF/1H split=all: 58747 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:44,478 INFO Loaded USDCHF/4H split=all: 15257 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:44,486 INFO Loaded USDCHF/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:44,487 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:44,563 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2214, 'BIAS_DOWN': 2634, 'BIAS_NEUTRAL': 3554}  ambiguous=3934 (total=8402)  short_runs_zeroed=549
2026-04-19 15:14:44,564 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:44,787 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 12380, 'RANGING': 9410, 'CONSOLIDATING': 4358, 'VOLATILE': 6593}  ambiguous=10596 (total=32741)  short_runs_zeroed=3896
2026-04-19 15:14:44,802 INFO Loaded USDCHF/5M split=train: 392805 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:45,062 INFO Loaded USDCHF/15M split=train: 130957 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:45,186 INFO Loaded USDCHF/1H split=train: 32741 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:45,284 INFO Loaded USDCHF/4H split=train: 8402 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:45,464 INFO Loaded USDJPY/5M split=all: 704798 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:45,465 INFO Loaded USDJPY/1H split=all: 58740 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:45,482 INFO Loaded USDJPY/4H split=all: 15254 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:45,491 INFO Loaded USDJPY/1D split=all: 2648 bars (2016-01-04 → 2026-02-27)
2026-04-19 15:14:45,492 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:45,579 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 2278, 'BIAS_DOWN': 2515, 'BIAS_NEUTRAL': 3610}  ambiguous=3986 (total=8403)  short_runs_zeroed=516
2026-04-19 15:14:45,581 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:45,816 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 13172, 'RANGING': 8282, 'CONSOLIDATING': 4697, 'VOLATILE': 6592}  ambiguous=10093 (total=32743)  short_runs_zeroed=4275
2026-04-19 15:14:45,831 INFO Loaded USDJPY/5M split=train: 392901 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:46,094 INFO Loaded USDJPY/15M split=train: 130972 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:46,222 INFO Loaded USDJPY/1H split=train: 32743 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:46,318 INFO Loaded USDJPY/4H split=train: 8403 bars (2016-01-04 → 2021-08-05)
2026-04-19 15:14:46,599 INFO Loaded XAUUSD/5M split=all: 1201053 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:14:46,601 INFO Loaded XAUUSD/1H split=all: 101228 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:14:46,618 INFO Loaded XAUUSD/4H split=all: 27183 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:14:46,630 INFO Loaded XAUUSD/1D split=all: 5296 bars (2009-03-15 → 2026-03-20)
2026-04-19 15:14:46,631 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:14:46,775 INFO Rule labels HTF_BIAS [4H]: {'BIAS_UP': 6629, 'BIAS_DOWN': 5241, 'BIAS_NEUTRAL': 7947}  ambiguous=8645 (total=19817)  short_runs_zeroed=980
2026-04-19 15:14:46,778 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:14:47,252 INFO Rule labels LTF_BEHAVIOUR [1H]: {'TRENDING': 29965, 'RANGING': 19624, 'CONSOLIDATING': 10050, 'VOLATILE': 14985}  ambiguous=23184 (total=74624)  short_runs_zeroed=9134
2026-04-19 15:14:47,298 INFO Loaded XAUUSD/5M split=train: 882017 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:14:47,816 INFO Loaded XAUUSD/15M split=train: 295079 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:14:48,012 INFO Loaded XAUUSD/1H split=train: 74624 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:14:48,145 INFO Loaded XAUUSD/4H split=train: 19817 bars (2009-03-15 → 2021-08-05)
2026-04-19 15:14:48,263 INFO train_multi: 44 segments, ~6069276 total bars
2026-04-19 15:14:48,263 INFO train_multi: training ALL 44 segments across TFs ['5M', '15M', '1H', '4H'] in one combined pass
2026-04-19 15:14:48,263 INFO train_multi: building combined dataset for TF=ALL (44 segments)
2026-04-19 15:19:17,446 INFO train_multi TF=ALL: 6067956 sequences across 44 segments
2026-04-19 15:19:17,446 INFO train_multi TF=ALL: estimated peak RAM = 10656 MB (train=479966 val=120011 n_feat=74 seq_len=30)
2026-04-19 15:19:18,760 INFO train_multi TF=ALL: train=479966 val=120011 (5335 MB tensors)
