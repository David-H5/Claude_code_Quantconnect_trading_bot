# System Architecture

## Overview

Event-driven microservices architecture with clear separation between data ingestion, signal generation, execution, and risk management.

```
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                        │
│  Claude-Flow Supervisor | Task Queue | State Management         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Market Data │  │  Strategy   │  │  Execution  │             │
│  │   Service   │──│   Engine    │──│   Service   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                      │
│                  ┌───────▼───────┐                              │
│                  │ Risk Manager  │                              │
│                  │  (Blocking)   │                              │
│                  └───────────────┘                              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  PostgreSQL (positions, trades, audit) | Redis (cache, pubsub) │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Market Data Service (`services/market_data/`)
- Real-time quotes via Schwab streaming API
- Options chain retrieval and IV surface calculation
- Greeks computation (delta, gamma, theta, vega)
- Historical data caching in Redis

### Strategy Engine (`algorithms/`)
- LEAN-based algorithm implementations
- Signal generation and scoring
- Backtest integration
- Paper trading validation layer

### Execution Service (`services/execution/`)
- Order management and routing
- Position tracking
- Fill reconciliation
- Slippage monitoring

### Risk Manager (`services/risk/`)
- Pre-trade validation (blocking)
- Position sizing calculations
- Exposure monitoring
- Kill switch implementation
- Circuit breaker patterns

## Data Flow

1. Market data streams into Redis pub/sub
2. Strategy engine subscribes, generates signals
3. Signals pass through Risk Manager validation
4. Approved orders sent to Execution Service
5. Execution confirms with broker, updates positions
6. All actions logged to PostgreSQL audit trail

## Key Files

| Path | Purpose |
|------|---------|
| `algorithms/base_options.py` | Base class for all options strategies |
| `services/risk/validator.py` | Pre-trade risk validation |
| `services/risk/kill_switch.py` | Emergency halt mechanism |
| `mcp/trading_server.py` | MCP server for Claude tool access |
| `config/limits.yaml` | Position and exposure limits |
