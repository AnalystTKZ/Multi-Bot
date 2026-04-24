# Training & Backtest Runbook

**Multi-Bot ICT/Smart Money Trading System — 2026-04-24**

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
# Expected: gru_lstm/model.pt  gru_lstm/temperature.pt  regime_htf.pkl  regime_ltf.pkl  quality_scorer.pkl  rl_ppo/model.zip  backups/
```

---

## Kaggle Training (Primary Path)

All PyTorch models are trained on Kaggle T4 × 2 GPUs using a split-dataset setup:

| Datasource | Path | Contents |
|---|---|---|
| Code (GitHub clone) | `/kaggle/input/datasets/tysonsiwela/multi-bot-system` | All Python code |
| Data | `/kaggle/input/datasets/tysonsiwela/trading-data` | `training_data/` + `processed_data/` (read-only) |
| Push clone | `/kaggle/working/remote/Multi-Bot` | Fresh git clone for pushing weights back to GitHub |

Run the full pipeline on Kaggle:

```python
# Cell 0 — setup
import os, subprocess
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
os.environ["GITHUB_TOKEN"] = token
# Clone fresh copy for GitHub push (separate from the code dataset)
subprocess.run(
    f"git clone https://{token}@github.com/AnalystTKZ/Multi-Bot.git /kaggle/working/remote/Multi-Bot",
    shell=True, check=True
)

# Cell 1 — run pipeline
%run /kaggle/working/Multi-Bot/trading-system/kaggle_train.py
```

Individual training steps:
```bash
python pipeline/step7_train.py     # GRU + Regime (GPU intensive)
python pipeline/step7b_train.py    # Quality + RL (uses journal)
python step8_push_to_github.py     # push weights + metrics to GitHub
```

**Important:** `step7_train.py` and `step7b_train.py` do NOT use `capture_output=True` — this prevents pipe-buffer deadlock when training output exceeds 64 KB.

**Latest training results (2026-04-24):**
- GRU: dir_acc ~0.58 (BCEWithLogitsLoss + temperature scaling), 74 features, 221,743 rows
- Regime 4H (HTF bias): 34 features, 3-class (BIAS_UP/DOWN/NEUTRAL) — BIAS_NEUTRAL recall known weakness (~30-38%)
- Regime 1H (LTF behaviour): 18 features, 4-class (TRENDING/RANGING/CONSOLIDATING/VOLATILE) — atr_pctile bug fixed
- Quality: dir_acc ~0.58, class-weighted Huber loss; labels read from real signal_metadata fields
- RL: warm-start from model.zip; needs ≥200 journal trades for action diversity

---

## Step 1 — Run the Offline Data Pipeline

> **Skip if `processed_data/` and `ml_training/datasets/` already exist.**

```bash
cd /home/tybobo/Desktop/Multi-Bot/trading-system

# Memory guard — required on local hardware (not needed on Kaggle)
ulimit -v 4000000

python3 kaggle_train.py
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

All models support **incremental warm-start**: existing weights are detected and training continues at 5× lower LR rather than reinitialising.

```bash
cd /home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine
source /home/tybobo/Desktop/Multi-Bot/.venv/bin/activate
export PYTHONPATH="/home/tybobo/Desktop/Multi-Bot/trading-system:/home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine"
ulimit -v 4000000
```

### 2a. Regime Classifier

- Input: OHLCV only — no journal needed
- HTF (4H): 34-feature MLP → 3-class (BIAS_UP/DOWN/NEUTRAL), mode="htf_bias"
- LTF (1H): 18-feature MLP → 4-class (TRENDING/RANGING/CONSOLIDATING/VOLATILE), mode="ltf_behaviour"
- Architecture: N → 128 → 64 → N_CLASSES with BatchNorm, GELU, residual skip, Dropout 0.5
- Labels: global GMM (`fit_global_gmm`) then rule-based labels with per-bar confidence weights
- Minimum persistence filter: short-run bars zeroed to reduce label noise
- Output: `weights/regime_htf.pkl`, `weights/regime_ltf.pkl`

```bash
python scripts/retrain_incremental.py --model regime
```

Expected: ≥65% HTF val accuracy (random = 33%), ≥55% LTF val accuracy (random = 25%).

### 2b. GRU-LSTM Predictor

- Input: OHLCV sequences (5M/15M/1H/4H) with regime context at each timestep
- Architecture: GRU(64, 2L) → LSTM(128, 2L) → shared FC(128→64) → 3 heads
- Loss: BCE(dir) + 0.5×SmoothL1(mag) + 0.3×NLL(var), dead-zone masking, label smoothing 0.05
- Temperature calibration: `fit_temperature()` saves `temperature.pt` after training
- Output: `weights/gru_lstm/model.pt` + `weights/gru_lstm/temperature.pt`
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

### 2c. Quality Scorer (EV Regressor)

- Input: `logs/trade_journal_detailed.jsonl` — needs ≥ 20 labelled trades with `exit_reason` in (`tp2`, `tp1`, `be_or_trail`, `sl_*`)
- Architecture: PyTorch MLP 17→64→32→1, class-weighted Huber loss
- Labels: tiered EV — `tp2=rr`, `tp1=0.75×rr`, `be_or_trail=0.4×rr`, `sl=-1.0`
- Output: `weights/quality_scorer.pkl`

```bash
python scripts/retrain_incremental.py --model quality
```

### 2d. RL Agent (PPO)

- Input: `logs/trade_journal_detailed.jsonl` — needs ≥ 500 records with `state_at_entry`
- Output: `weights/rl_ppo/model.zip` (explicit `.zip` extension — critical)
- Device: CPU always (MLP policy)

```bash
python scripts/retrain_incremental.py --model rl
```

### 2e. Train All

```bash
python scripts/retrain_incremental.py
```

Order: regime → gru → quality → rl → sentiment.

---

## Step 3 — Verify All Weights Exist

```bash
python3 -c "
import os
weights = {
    'GRU-LSTM':      'weights/gru_lstm/model.pt',
    'GRU-Temp':      'weights/gru_lstm/temperature.pt',
    'Regime HTF':    'weights/regime_htf.pkl',
    'Regime LTF':    'weights/regime_ltf.pkl',
    'QualityScorer': 'weights/quality_scorer.pkl',
    'RLAgent':       'weights/rl_ppo/model.zip',
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

## Step 4 — Run the Backtest

```bash
cd /home/tybobo/Desktop/Multi-Bot/trading-system/trading-engine
python scripts/run_backtest.py
```

Results saved to `backtest_results/backtest_YYYYMMDD_HHMMSS.json`.

**Latest results (2026-04-24, train 2016-2021, val 2021-2023, test 2023-2025, $10k capital):**

| Window | WR | PF | Sharpe | MaxDD | Notes |
|---|---|---|---|---|---|
| Val (2021–2023) | 53.8–54.3% | 2.97–3.14 | 5.11–5.29 | ≤2.5% | Across 3 rounds |
| Blind test (2023–2025) | 50.9–51.4% | 2.45–2.52 | 4.83–4.96 | ≤2.9% | True OOS |
| Post-retrain 3yr | 54.6% WR | — | — | 2.8% | 3826% return R1 |

Data: 221,743 rows, 202 features, 11 symbols (AUDUSD/EURGBP/EURJPY/EURUSD/GBPJPY/GBPUSD/NZDUSD/USDCAD/USDCHF/USDJPY/XAUUSD)
Regime distribution uses current class names: HTF (BIAS_UP/DOWN/NEUTRAL), LTF (TRENDING/RANGING/CONSOLIDATING/VOLATILE)

---

## Step 5 — Start the Live Engine

```bash
cd /home/tybobo/Desktop/Multi-Bot/trading-system
docker compose up -d
```

Check engine started with ML active:
```bash
docker compose logs trading-engine --tail=30
# Should see: "ML_ENABLED=True — loading models" and each model loading

docker exec trading_backend curl -s http://trading-engine:8000/health
```

---

## Step 6 — Retrain Schedule

`retrain_scheduler.py` in the `trading_model_retrainer` container manages two schedules:

| Cadence | Models | Schedule | Min journal entries |
|---|---|---|---|
| Weekly | Quality + RL | Sunday 02:00 UTC | 200 (Quality), 500 (RL) |
| Monthly | Regime + GRU | 1st Sunday 03:00 UTC | OHLCV only |

All retrains are warm-start.

Manual trigger:
```bash
docker exec trading_model_retrainer python /app/scripts/retrain_incremental.py --model quality
docker exec trading_model_retrainer python /app/scripts/retrain_incremental.py --model rl
docker exec trading_model_retrainer python /app/scripts/retrain_incremental.py --model regime
docker exec trading_model_retrainer python /app/scripts/retrain_incremental.py --model gru
```

---

## Troubleshooting

### `ModelNotTrainedError` at engine startup
```bash
docker exec trading_engine_main python /app/scripts/retrain_incremental.py --dry-run
docker exec trading_engine_main python /app/scripts/retrain_incremental.py --model <name>
```

### `RuntimeError: ml_predictions missing 'ev' key`
QualityScorer trained but EV output not reaching Guard 7. Check `_run_ml_inference()` — verify `ev` / `quality_score` keys are included in the returned dict.

### `RLAgent.load failed: [Errno 21] Is a directory`
Old weights saved as a directory named `model`. Delete `weights/rl_ppo/model` (the directory) and retrain. The fixed code now saves as `model.zip`.

### `RuntimeError: ml_predictions missing 'regime' key`
RegimeClassifier failed. Retrain:
```bash
docker exec trading_engine_main python /app/scripts/retrain_incremental.py --model regime
```

### Round 2/3 trade count drops to 0
GRU included in per-round retrain loop — causes catastrophic forgetting. Verify `step6_backtest.py` only retrains `["regime", "quality", "rl"]`.

### Journal EV fields are null
The `exit_reason` in backtest was being collapsed to `tp/sl` before the journal write, breaking `_compute_ev_label`. Fixed in `step6_backtest.py` — ensure you have the latest version.

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
python kaggle_train.py

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

# Trade journal (live)
tail -f trading-engine/logs/trade_journal_detailed.jsonl | python -m json.tool
```
