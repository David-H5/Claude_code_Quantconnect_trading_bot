# Commands Reference

## Development

```bash
make dev              # Start local development environment
make shell            # Enter Docker development container
make logs             # Tail all service logs
```

## Build & Test

```bash
make build            # Build all services
make test             # Run full test suite
make test-unit        # Unit tests only
make test-integration # Integration tests only
make lint             # Run ruff + black + isort
make typecheck        # Run mypy strict mode
make coverage         # Generate coverage report (target: 80%+)
```

## LEAN / QuantConnect

```bash
lean init                           # Initialize LEAN project
lean backtest "AlgorithmName"       # Run local backtest
lean backtest "AlgorithmName" --debug  # Backtest with debugging
lean cloud backtest "AlgorithmName" --push  # Push and run on QC cloud
lean live deploy "AlgorithmName" --brokerage "Paper Trading"  # Paper trade
lean live deploy "AlgorithmName" --brokerage "Charles Schwab"  # LIVE - requires approval
lean data download --dataset "US Equity Options"  # Download options data
```

## Database

```bash
make db-migrate       # Run database migrations
make db-rollback      # Rollback last migration
make db-seed          # Seed development data
make db-reset         # Drop and recreate database
```

## Docker

```bash
make docker-build     # Build all containers
make docker-up        # Start all services
make docker-down      # Stop all services
make docker-sandbox   # Start isolated sandbox for autonomous work
```

## Hooks

```bash
python .claude/hooks/risk_validator.py --test  # Test risk validation
python .claude/hooks/log_trade.py --test       # Test trade logging
```

## MCP Servers

```bash
claude mcp list                                    # List configured servers
claude mcp add market-data -- python mcp/market_data_server.py  # Add server
claude mcp remove market-data                      # Remove server
```
