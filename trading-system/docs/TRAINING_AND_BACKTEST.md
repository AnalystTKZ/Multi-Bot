# Training & Backtest Runbook

**Multi-Bot ICT/Smart Money Trading System — 2026-04-18**

`ML_ENABLED=true` is the default. This document is the single authoritative guide for training all models and running a backtest.

---

## Prerequisites

```bash
# Virtual environment (Python 3.12)
source /home/tybobo/Desktop/Multi-Bot/.venv/bin/activate

# Set PYTHONPATH so both the pipeline and engine imports resolve
export PYTHONPATH="/home/tybobo/Desktop/Multi-Bot/trading-system:/home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine"

# All training commands below must be run from this directory
cd /home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine
```

Confirm weights directory exists:

```bash
ls weights/
# Expected: gru_lstm/model.pt  regime_4h.pkl  regime_1h.pkl  quality_scorer.pkl  rl_ppo/  backups/
```

---

## Kaggle Training (Primary Path)

All PyTorch models are trained on Kaggle T4 × 2 GPUs. The pipeline triggers training automatically. Run on Kaggle:

```bash
python run_pipeline.py   # runs all steps 0–8 including training + GitHub push
```

Individual training steps:
```bash
python pipeline/step7_train.py     # GRU + Regime (GPU intensive)
python pipeline/step7b_train.py    # Quality + RL (uses 8,203 journal trades)
python step8_push_to_github.py     # push weights + metrics to GitHub
```

**Important:** `step7_train.py` and `step7b_train.py` do NOT use `capture_output=True` — this prevents pipe-buffer deadlock when training output exceeds 64 KB.

**Latest training results (2026-04-18):**
- GRU: 7,452,801 samples, 44 combos, 30 val loss checkpoints
- Regime 4H: 101,416 train / 25,354 val, accuracy 48.8%
- Regime 1H: 394,124 train / 98,531 val, accuracy 41.1%
- Quality + RL: 8,203 journal entries (step 7b)

---

## Step 1 — Run the Offline Data Pipeline

> **Skip if `processed_data/` and `ml_training/datasets/` already exist.**

```bash
cd /home/tybobo/Desktop/Multi-Bot/trading-system

# Memory guard — required on local hardware (not needed on Kaggle)
ulimit -v 4000000

python3 run_pipeline.py
```

Individual steps:

```bash
python3 pipeline/step0_resample.py      # M1 → 5 MTF parquets per symbol
python3 pipeline/step1_inventory.py     # data discovery
python3 pipeline/step2_clean.py         # clean + merge macro columns
python3 pipeline/step3_align.py         # multi-asset alignment
python3 pipeline/step4_features.py      # 200+ feature engineering passes
python3 pipeline/step5_split.py         # 70/15/15 time-based split
python3 pipeline/step6_backtest.py      # 3-round reinforcement backtest
python3 pipeline/step7_train.py         # GRU + Regime (Kaggle GPU)
python3 pipeline/step7b_train.py        # Quality + RL
python3 step8_push_to_github.py         # push weights + metrics to GitHub
```

---

## Step 2 — Train the ML Models

All models support **incremental warm-start**: existing weights are detected and training continues at 5× lower LR rather than reinitialising. This preserves knowledge from 7+ years of historical data.

```bash
cd /home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine
source /home/tybobo/Desktop/Multi-Bot/.venv/bin/activate
export PYTHONPATH="/home/tybobo/Desktop/Multi-Bot/trading-system:/home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine"
ulimit -v 4000000
```

### 2a. Regime Classifier (PyTorch MLP, GPU)

- Input: OHLCV only — no journal needed
- Architecture: 59 → 128 → 64 → 4, BatchNorm, GELU, residual skip
- Labels: GMM clustering per symbol group (dollar/cross/gold) on 4H features
- Output: `weights/regime_4h.pkl`, `weights/regime_1h.pkl`
- Warm-start: preserves weights if feature count matches

```bash
python scripts/retrain_incremental.py --model regime
```

Expected accuracy: 40–55% on val set (4-class balanced; random = 25%).

### 2b. GRU-LSTM Predictor (PyTorch, GPU)

- Input: OHLCV sequences (5M/15M/1H/4H timeframes)
- Architecture: GRU(256, 2L) + LayerNorm → direction head + magnitude head + variance head
- Labels: 12-bar forward log return, dead-zone ATR masking, label smoothing 0.05
- Output: `weights/gru_lstm/model.pt`
- **Not warm-started from journal** — retrains from full OHLCV history only (monthly)

```bash
python scripts/retrain_incremental.py --model gru
```

Verify:
```bash
python3 -c "
import torch
ckpt = torch.load('weights/gru_lstm/model.pt', map_location='cpu')
print('GRU loaded. Keys:', list(ckpt.keys())[:5])
"
```

### 2c. Quality Scorer (PyTorch MLP, GPU)

- Input: `logs/trade_journal_detailed.jsonl` — needs ≥ 20 labelled trades (TP or SL outcome)
- Architecture: 17 → 64 → 32 → 1, BatchNorm, GELU, Huber loss
- Labels: `rr_ratio` directly (wins normalised by median winner RR)
- Loss: class-weighted Huber (~4.6× on winners) to correct 18%/82% imbalance
- Output: `weights/quality_scorer.pkl`
- Warm-start: detects existing weights, trains at `lr=2e-4` vs `lr=1e-3`

```bash
python scripts/retrain_incremental.py --model quality
```

> **First run bootstrap:** `step7b_train.py` bootstraps QualityScorer from the backtest trade log.

### 2d. RL Agent (PPO via stable-baselines3)

- Input: `logs/trade_journal_detailed.jsonl` — needs ≥ 500 episode records with `state_at_entry`
- Output: `weights/rl_ppo/`
- Warm-start: `set_env()` + `learning_rate = 3e-4 / 5.0` + `reset_num_timesteps=False`

```bash
python scripts/retrain_incremental.py --model rl
```

### 2e. Sentiment Model (FinBERT — pre-trained, no training needed)

```bash
python scripts/retrain_incremental.py --model sentiment
```

### 2f. Train All

```bash
python scripts/retrain_incremental.py
```

Trains in order: regime → gru → quality → rl → sentiment.

---

## Step 3 — Verify All Weights Exist

```bash
python3 -c "
import os
weights = {
    'GRU-LSTM':         'weights/gru_lstm/model.pt',
    'Regime 4H':        'weights/regime_4h.pkl',
    'Regime 1H':        'weights/regime_1h.pkl',
    'QualityScorer':    'weights/quality_scorer.pkl',
    'RLAgent':          'weights/rl_ppo',
}
all_ok = True
for name, path in weights.items():
    exists = os.path.exists(path)
    print(f'  [{\"OK\" if exists else \"MISSING\"}] {name}: {path}')
    if not exists: all_ok = False
print()
print('All weights present' if all_ok else 'MISSING WEIGHTS — retrain before starting engine')
"
```

---

## Step 4 — Run the Backtest (with ML_ENABLED=true)

```bash
cd /home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine
python scripts/run_backtest.py
```

Default date range: 2019-01-01 to 2025-12-31. Results saved to `backtest_results/backtest_YYYYMMDD_HHMMSS.json`.

Specific traders / date range:
```bash
python scripts/run_backtest.py 1 3 5
python scripts/run_backtest.py --start 2022-01-01 --end 2024-12-31
```

Read output:
```bash
cat backtest_results/backtest_YYYYMMDD_HHMMSS.json | python3 -m json.tool | head -80
```

**Latest results (2026-04-18):**
- Round 1: 2,712 trades, WR 45.5%, PF 1.99, Sharpe 3.50, MaxDD 2.72%, Return +20.1%
- Round 3: 2,769 trades, WR 44.9%, PF 1.93, Sharpe 3.40, MaxDD 2.45%, Return +18.0%
- High-conviction: confidence ≥ 0.90 → 67% WR on 291 trades

---

## Step 5 — Start the Live Engine

After all weights are verified:

```bash
cd /home/tybobo/Desktop/Multi-Bot/trading-system
docker compose up -d
```

Check engine started with ML active:
```bash
docker compose logs trading-engine --tail=30
# Should see: "ML_ENABLED=True — loading models" and each model loading
# No ModelNotTrainedError lines

docker exec trading_backend curl -s http://trading-engine:8000/health
```

---

## Step 6 — Retrain Schedule

`retrain_scheduler.py` in the `trading_model_retrainer` container manages two separate schedules:

| Cadence | Models | Schedule | Min journal entries |
|---|---|---|---|
| Weekly | Quality + RL | Sunday 02:00 UTC | 200 (Quality), 500 (RL) |
| Monthly | Regime + GRU | 1st Sunday 03:00 UTC | OHLCV only |

All retrains are warm-start — no reinitialisation from scratch.

Trigger manually:
```bash
docker exec trading_engine_main python /app/scripts/retrain_incremental.py --model quality
docker exec trading_engine_main python /app/scripts/retrain_incremental.py --model rl
docker exec trading_engine_main python /app/scripts/retrain_incremental.py --model regime
docker exec trading_engine_main python /app/scripts/retrain_incremental.py --model gru
```

---

## Troubleshooting

### `ModelNotTrainedError` at engine startup
```bash
docker exec trading_engine_main python /app/scripts/retrain_incremental.py --dry-run
docker exec trading_engine_main python /app/scripts/retrain_incremental.py --model <name>
```

### `RuntimeError: ml_predictions missing 'ev' key`
QualityScorer trained but output not reaching Guard 7. Check `_run_ml_inference()` — verify `ev` / `quality_score` keys are included in the returned dict.

### `RuntimeError: ml_predictions missing 'regime' key`
Fires inside Trader 1, 3, or 5 when `ML_ENABLED=true` but RegimeClassifier failed. Retrain:
```bash
docker exec trading_engine_main python /app/scripts/retrain_incremental.py --model regime
```

### Round 2/3 trade count drops to 0
GRU was incorrectly included in the per-round retrain loop — fine-tuning on ~3k journal trades causes catastrophic forgetting. Verify `step6_backtest.py` only retrains `["regime", "quality", "rl"]`, not `"gru"`.

### Docker daemon stale lock
```bash
pkill -u $USER containerd; pkill -u $USER dockerd
rm ~/.local/share/docker/containerd/daemon/io.containerd.metadata.v1.bolt/meta.db
systemctl --user reset-failed docker && systemctl --user start docker
```

---

## Quick Reference

```bash
# Kaggle: full pipeline
python run_pipeline.py

# Local: train all models
cd /home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine && \
  source /home/tybobo/Desktop/Multi-Bot/.venv/bin/activate && \
  export PYTHONPATH="/home/tybobo/Desktop/Multi-Bot/trading-system:/home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine" && \
  ulimit -v 4000000 && \
  python scripts/retrain_incremental.py

# Backtest
python scripts/run_backtest.py

# Start all containers
cd /home/tybobo/Desktop/Multi-Bot/trading-system && docker compose up -d

# Logs
docker compose logs trading-engine --tail=50 -f

# Trade journal
tail -f trading-engine/logs/trade_journal_detailed.jsonl | python -m json.tool
```
