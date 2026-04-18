# Multi-Bot ICT Smart Money Trading System

A production-grade automated trading system implementing Inner Circle Trader (ICT) and Smart Money Concepts (SMC) across multiple asset classes.

## Overview

This system deploys 6 independent trading bots:

- **4 Trader Bots**: EMA Crossover, Mean Reversion, Breakout, News-Driven
- **2 Monitor Bots**: Chart Surveillance, News Sentiment Tracking

## Architecture

- **Frontend**: React.js dashboard for monitoring and control
- **Backend**: FastAPI service for REST + monitoring endpoints
- **Trading Engine**: Python microservices for bot logic and execution
- **Data Layer**: PostgreSQL, MongoDB, InfluxDB for persistence
- **Infrastructure**: Docker containers orchestrated with Docker Compose

## Key Features

- ICT/SMC strategy implementation (Order Blocks, FVGs, Liquidity Sweeps)
- Multi-asset support (Forex, Commodities, Crypto)
- Risk management with position locking and drawdown controls
- Real-time monitoring and alerting
- Backtesting framework for strategy validation
- Production-ready with logging, error handling, and health checks

## Quick Start

1. Clone the repository
2. Configure environment variables (copy `trading-system/.env.example` to `trading-system/.env`)
3. Run `docker compose -f docker-compose.dev.yml up -d --build`
4. Access the dashboard at http://localhost:3001

## Documentation

See [docs/](docs/) for detailed documentation.

## Disclaimer

This is a trading system for educational and research purposes. Use at your own risk. Past performance does not guarantee future results.
