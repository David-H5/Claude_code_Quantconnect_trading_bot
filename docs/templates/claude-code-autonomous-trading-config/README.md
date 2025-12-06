# Claude Code Autonomous Trading Bot Configuration

Template files for configuring Claude Code for autonomous trading bot development.

**Source**: Claude Browser advice + PDF Guide
**Date Received**: December 2025
**Analysis**: See [FEATURE_COMPARISON_ANALYSIS.md](FEATURE_COMPARISON_ANALYSIS.md)

## Contents

### Core Configuration

| File | Purpose |
|------|---------|
| [CLAUDE.md](CLAUDE.md) | Root project instructions with @import pattern |
| [settings.json](settings.json) | Hooks, permissions, and environment config |
| [.mcp.json](.mcp.json) | MCP server configuration (4 servers) |
| [Makefile](Makefile) | Unified build/test/deploy commands |

### Documentation (for @import)

| File | Purpose |
|------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Microservices architecture diagram |
| [SAFETY.md](SAFETY.md) | Trading limits, kill switch, circuit breaker |
| [WORKFLOWS.md](WORKFLOWS.md) | Development patterns (explore→plan→code→test) |
| [COMMANDS.md](COMMANDS.md) | Quick reference for common commands |

### Agent Personas (.claude/agents/)

| File | Purpose | Tools |
|------|---------|-------|
| [senior-engineer.md](senior-engineer.md) | Production-quality code | Full |
| [risk-reviewer.md](risk-reviewer.md) | Risk/compliance review | Read-only |
| [strategy-dev.md](strategy-dev.md) | Algorithm development | Full |
| [code-review.md](code-review.md) | PR review | Read-only |
| [qa-engineer.md](qa-engineer.md) | Testing specialist | Full |
| [researcher.md](researcher.md) | Research tasks | Read + Web |
| [backtest-analyst.md](backtest-analyst.md) | Results analysis | Read-only |

### Hook Scripts (.claude/hooks/)

| File | Purpose |
|------|---------|
| [risk_validator.py](risk_validator.py) | PreToolUse: Block orders exceeding limits |
| [log_trade.py](log_trade.py) | PostToolUse: Audit trail for broker calls |
| [load_context.py](load_context.py) | SessionStart: Inject portfolio state |

### MCP Servers (mcp/)

| File | Purpose |
|------|---------|
| [market_data_server.py](market_data_server.py) | Quotes, options chains, Greeks, historical |

## Usage

These are **reference templates**. To apply:

1. **Review** [FEATURE_COMPARISON_ANALYSIS.md](FEATURE_COMPARISON_ANALYSIS.md) for prioritized upgrades
2. **Copy** files you want to use to appropriate locations
3. **Adapt** to match your existing patterns
4. **Test** in a new Claude Code session

### Quick Start

```bash
# Copy hook scripts
cp docs/templates/claude-code-autonomous-trading-config/risk_validator.py .claude/hooks/
cp docs/templates/claude-code-autonomous-trading-config/load_context.py .claude/hooks/

# Create agents folder and copy personas
mkdir -p .claude/agents
cp docs/templates/claude-code-autonomous-trading-config/*-*.md .claude/agents/

# Merge .mcp.json into project root (review first!)
```

### Important Notes

- **CLAUDE.md files here are INACTIVE** - they don't affect Claude Code behavior
- **To activate**: merge sections into root `/CLAUDE.md` or move to appropriate location
- **Test changes** in a new Claude Code session to verify behavior
