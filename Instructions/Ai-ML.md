# ALLOWED ARCHITECTURE
┌─────────────────────────────────────────┐
│  Trading Bot System                     │
├─────────────────────────────────────────┤
│                                         │
│  Market Data Sources                    │
│  ├─ Broker APIs (Real prices)          │
│  ├─ News APIs (Real news)              │
│  └─ Economic Calendar (Real events)    │
│           ↓                             │
│  ML/AI Models (Trained on market data) │
│  ├─ Price Prediction (LSTM)            │
│  ├─ Sentiment Analysis (NLP)           │
│  ├─ Pattern Recognition (CNN)          │
│  ├─ Anomaly Detection (Isolation)      │
│  └─ RL Agent (PPO/DQN)                 │
│           ↓                             │
│  Trading Signals                        │
│  ├─ ICT/SM Signals (Rule-based)        │
│  ├─ ML Predictions                     │
│  └─ Ensemble (Combined)                │
│           ↓                             │
│  Position Management                    │
│  ├─ Risk Management                    │
│  ├─ Conflict Resolution                │
│  └─ Order Execution                    │
│                                         │               │
│  (No Claude outputs used as training)  │
│                                         │
└─────────────────────────────────────────┘