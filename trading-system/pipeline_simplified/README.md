# Simplified Pipeline

Parallel MVP pipeline for `simplify-v2`. It does not replace the canonical
`pipeline/` runner or its artifacts.

```bash
cd trading-system
python pipeline_simplified/run_simple_pipeline.py --list
python pipeline_simplified/run_simple_pipeline.py --force
```

Outputs:

- `processed_data/simple/ohlcv/`
- `processed_data/simple/features.parquet`
- `ml_training/simple_datasets/`
- `trading-engine/weights/unified_direction_regime/model.pt`

The live engine can opt into the simplified model with:

```bash
SIMPLIFIED_ML_ENABLED=true SIMPLIFIED_USE_QUALITY=false python trading-engine/main.py
```

Default symbol scope is fixed to `XAUUSD,EURUSD,USDJPY,EURJPY,GBPJPY,GBPUSD`.
