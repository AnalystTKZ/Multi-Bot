# Kaggle Simplified Pipeline

Run from the Kaggle checkout root:

```bash
cd /kaggle/working/Multi-Bot/trading-system

export SIMPLE_SYMBOLS=XAUUSD,EURUSD,USDJPY,EURJPY,GBPJPY,GBPUSD
export SIMPLE_START_DATE=2020-01-01
export UNIFIED_EPOCHS=12
export UNIFIED_BATCH_SIZE=2048

python pipeline_simplified/run_simple_pipeline.py --force
```

Outputs:

- `processed_data/simple/ohlcv/`
- `processed_data/simple/features.parquet`
- `ml_training/simple_datasets/{train,validation,test}.parquet`
- `ml_training/simple_metrics/unified_training_summary.json`
- `trading-engine/weights/unified_direction_regime/model.pt`

Backtest with the simplified model:

```bash
export SIMPLIFIED_ML_ENABLED=true
export ML_ENABLED=true
export BT_WINDOW=round1

python pipeline/step6_backtest.py
```

The old FAISS/vector-store and sentiment model paths have been removed. The active
symbol scope is fixed to:

```text
XAUUSD, EURUSD, USDJPY, EURJPY, GBPJPY, GBPUSD
```
