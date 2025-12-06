# Autonomous Options Trading Bot

QuantConnect LEAN-based autonomous options trading system with Charles Schwab API integration. Claude Code powers development automation with safety-first architecture.

## Quick Reference

See @docs/COMMANDS.md for build/test/deploy commands.
See @docs/ARCHITECTURE.md for system design and component overview.
See @docs/SAFETY.md for trading constraints and risk controls.
See @docs/WORKFLOWS.md for development patterns and task flows.

## Tech Stack

- **Engine**: QuantConnect LEAN (Python 3.11)
- **Broker**: Charles Schwab API via schwab-py
- **Backend**: FastAPI, PostgreSQL, Redis
- **Infrastructure**: Docker, GitHub Actions
- **Testing**: pytest, backtrader for strategy validation

## Project Structure

```
trading-bot/
├── algorithms/          # LEAN trading algorithms
├── services/            # Microservices (data, execution, risk)
├── mcp/                 # Custom MCP servers for market data
├── tests/               # Unit, integration, backtest suites
├── backtests/           # Backtest results and analysis
├── .claude/             # Claude Code configuration
│   ├── commands/        # Slash commands
│   ├── agents/          # Subagent definitions
│   ├── hooks/           # Hook scripts
│   └── settings.json    # Hooks and permissions
└── docs/                # Documentation (imported into context)
```

## CRITICAL CONSTRAINTS

**YOU MUST follow these rules without exception:**

1. NEVER execute live trades without explicit user confirmation
2. NEVER modify files in `algorithms/live/` without running paper validation first
3. NEVER commit API keys, tokens, or secrets to git
4. ALWAYS run `make test` before committing any algorithm changes
5. ALWAYS validate position sizing against limits in @docs/SAFETY.md
6. STOP and ask if uncertain about any trading logic

## DO NOT

- Edit `.env` or `.env.local` files directly
- Modify `algorithms/production/` without explicit approval
- Skip backtesting for any strategy changes
- Disable or bypass risk validation hooks
- Use `--dangerously-skip-permissions` outside Docker sandbox

## Verification

After any code change:
1. Run `make lint` - fix all issues
2. Run `make test` - ensure 100% pass
3. Run `make typecheck` - resolve type errors
4. For algorithm changes: `lean backtest <algorithm>` minimum 100 trades
