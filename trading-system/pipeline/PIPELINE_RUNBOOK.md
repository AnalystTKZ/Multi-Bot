cd /home/tybobo/Desktop/Multi-Bot/trading-system && export PYTHONPATH="/home/tybobo/Desktop/Multi-Bot/trading-system:/home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine" && python3 run_pipeline.py 2>&1 | tee -a pipeline_run.log

# Pipeline Runbook

Complete data pipeline, backtesting, and ML training for the Multi-Bot ICT trading system.

---

## Prerequisites

```bash
cd /home/tybobo/Desktop/Multi-Bot/trading-system

export PYTHONPATH="/home/tybobo/Desktop/Multi-Bot/trading-system:/home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine"
```

> Re-run the `export` line every new terminal session before running any step.

---

## Memory Notes

- Run **one step at a time** — wait for each to finish before starting the next
- No parallel or background processes
- If a step crashes with OOM, prepend `ulimit -v 3000000` (caps virtual memory at ~3 GB)
 - The pipeline now **cleans previous outputs by default** so it starts fresh each run.
   To keep prior outputs, pass `--no-clean`.

```bash
ulimit -v 3000000 && python3 pipeline/stepN_name.py
```

---

## Steps

### Step 1 — Data Inventory

Scans all `training_data/` files. Writes `processed_data/raw_inventory.json`.

```bash
python3 pipeline/step1_inventory.py
```

**Check:**
```bash
ls -lh processed_data/raw_inventory.json
```

---

### Step 2 — Data Cleaning

Cleans and normalizes each symbol one at a time. Writes per-symbol parquet files to `processed_data/clean/`.

```bash
python3 pipeline/step2_clean.py
```

**Check:**
```bash
ls -lh processed_data/clean/
```

---

### Step 3 — Multi-Domain Alignment

Aligns all symbols to a single 15M timeline. Adds cross-asset context. Writes `processed_data/aligned_multi_asset.parquet`.

```bash
python3 pipeline/step3_align.py
```

**Check:**
```bash
ls -lh processed_data/aligned_multi_asset.parquet
cat processed_data/alignment_summary.json
```

---

### Step 4 — Feature Engineering

Adds technical indicators (EMA, RSI, ATR, ADX, BB), SMC features (BOS, FVG, sweeps, order blocks), cross-asset features, fundamental features, session flags, and RL-state compatible features. All in-place — no extra copies. Writes `processed_data/feature_engineered.parquet`.

```bash
python3 pipeline/step4_features.py
```

**Check:**
```bash
ls -lh processed_data/feature_engineered.parquet
cat processed_data/feature_manifest.json
```

---

### Step 5 — Train / Val / Test Split

Strict time-based split — **no shuffling**.

| Split      | Ratio |
|------------|-------|
| Train      | 70%   |
| Validation | 15%   |
| Test       | 15%   |

Writes to `ml_training/datasets/`.

```bash
python3 pipeline/step5_split.py
```

**Check:**
```bash
ls -lh ml_training/datasets/
cat ml_training/datasets/split_summary.json
```

---

### Step 6 — Backtest (All 5 Traders)

Runs `trading-engine/scripts/run_backtest.py` for traders 1–5 across all supported symbols. Results saved to `backtesting/results/`.

```bash
python3 pipeline/step6_backtest.py
```

**Check:**
```bash
ls -lh backtesting/results/
cat backtesting/results/latest_summary.json
```

---

### Step 7 — Model Training

Generates synthetic trade journal if empty, then runs `trading-engine/scripts/retrain_incremental.py` for each model in order:

1. `regime` — LightGBM regime classifier
2. `gru` — GRU-LSTM price predictor
3. `quality` — XGBoost quality scorer
4. `rl` — PPO RL agent

Weights saved to `trading-engine/models/weights/`. Copies to `ml_training/models/`.

```bash
python3 pipeline/step7_train.py
```

**Check:**
```bash
ls -lh trading-engine/models/weights/
cat ml_training/metrics/training_summary.json
```

---

### Step 8 — Validation + Critic Report

Evaluates trained models on the test split. Compiles backtest metrics. Produces a full critic report.

```bash
python3 pipeline/step8_validate.py
```

**Check:**
```bash
cat ml_training/metrics/critic_report.json
```

---

## Output Structure

```
processed_data/
├── raw_inventory.json
├── cleaning_summary.json
├── alignment_summary.json
├── feature_manifest.json
├── clean/
│   ├── EURUSD_15M.parquet
│   ├── GBPUSD_15M.parquet
│   ├── USDJPY_15M.parquet
│   ├── AUDUSD_15M.parquet
│   ├── USDCAD_15M.parquet
│   └── XAUUSD_15M.parquet
├── aligned_multi_asset.parquet
└── feature_engineered.parquet

ml_training/
├── datasets/
│   ├── train.parquet
│   ├── validation.parquet
│   ├── test.parquet
│   └── split_summary.json
├── models/
│   ├── gru_lstm/
│   ├── regime_classifier.pkl
│   ├── quality_scorer.pkl
│   └── rl_ppo/
├── metrics/
│   ├── training_summary.json
│   ├── retrain_history.json
│   └── critic_report.json
└── logs/

backtesting/
├── datasets/
├── results/
│   ├── backtest_YYYYMMDD_HHMMSS.json
│   └── latest_summary.json
└── logs/
```

---

## After All Steps Pass

Enable ML in the trading engine:

```bash
# In trading-system/.env
ML_ENABLED=true
```

Then restart the engine:

```bash
cd /home/tybobo/Desktop/Multi-Bot/trading-system
docker compose restart trading-engine
docker compose logs trading-engine --tail=50 -f
```

---

## Re-running a Single Step

Each step is idempotent for most cases. Step 2 skips already-cleaned symbols. To force a re-run, delete the output first:

```bash
# Re-run step 2 for one symbol
rm processed_data/clean/EURUSD_15M.parquet
python3 pipeline/step2_clean.py

# Re-run step 4 from scratch
rm processed_data/feature_engineered.parquet
python3 pipeline/step4_features.py
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: indicators` | Re-export `PYTHONPATH` (see Prerequisites) |
| `aligned_multi_asset.parquet not found` | Run step 3 first |
| `feature_engineered.parquet not found` | Run step 4 first |
| `ModelNotTrainedError` at runtime | Run step 7, then set `ML_ENABLED=true` |
| Step crashes with OOM | `ulimit -v 3000000` before the command |
| Step 6 — zero trades for trader_4 | Expected — News Momentum needs live news feed |
| `index.lock` git error | `rm /home/tybobo/Desktop/Multi-Bot/.git/index.lock` |
