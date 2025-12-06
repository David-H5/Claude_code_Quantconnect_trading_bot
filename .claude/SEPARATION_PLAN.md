# Trading Framework / Development Tools Separation Plan

**Created:** 2025-12-05
**Updated:** 2025-12-06
**Status:** ⏸️ DEFERRED - See NEXT_REFACTOR_PLAN.md Phase 6 Analysis
**Prerequisite:** ✅ REFACTOR_PLAN.md Phase 1-2 COMPLETE

> **⚠️ DEFERRED:** This plan conflicts with the 5-layer architecture established in
> NEXT_REFACTOR_PLAN.md Phase 5. The layer architecture already provides clear module
> boundaries without physical file moves. See NEXT_REFACTOR_PLAN.md lines 480-535 for
> the detailed analysis of why this was deferred (334+ import changes, circular import
> risk, Layer 0-1 modules classified as dev tools but needed by trading code).

---

## Research Sources

This plan is based on industry best practices from:
- [Tweag: Python Monorepo Structure](https://www.tweag.io/blog/2023-04-04-python-monorepo-1/)
- [The Hitchhiker's Guide to Python: Project Structure](https://docs.python-guide.org/writing/structure/)
- [Alpaca: Building Algorithmic Trading Bots](https://alpaca.markets/learn/algorithmic-trading-bot-7-steps)
- [Real Python: Separating Dev and Prod Dependencies](https://realpython.com/lessons/separating-development-and-production-dependencies/)
- [Dagster: How to Structure Python Projects](https://dagster.io/blog/python-project-best-practices)

---

## Executive Summary

The project currently mixes **production trading code** with **Claude Code development automation**. This creates:
1. Confusion about what's "the product" vs "development tools"
2. Risk of development tools affecting production trading
3. Difficulty deploying trading code independently
4. Unclear ownership boundaries

**Solution:** Adopt a `src/` layout that cleanly separates:
- `src/trading/` - Production trading infrastructure (deployable)
- `src/claude_dev/` - Claude Code development tools (not deployed)

---

## Current vs Proposed Structure

### Current Structure (Mixed)

```
project/
├── .claude/              # Dev tools (Claude Code)
├── algorithms/           # TRADING
├── agents_deprecated/    # DEPRECATED
├── analytics/            # TRADING
├── api/                  # TRADING
├── backtesting/          # TRADING
├── compliance/           # TRADING
├── config/               # MIXED
├── data/                 # TRADING
├── docs/                 # DOCS
├── evaluation/           # DEV TOOLS
├── execution/            # TRADING
├── indicators/           # TRADING
├── infrastructure/       # TRADING
├── integration_tests/    # TESTING
├── llm/                  # TRADING (LLM agents)
├── logs/                 # OUTPUT
├── mcp/                  # TRADING
├── models/               # TRADING
├── observability/        # MIXED
├── prompts/              # TRADING (LLM prompts)
├── reasoning_chains/     # TRADING (LLM)
├── research/             # DOCS
├── scanners/             # TRADING
├── scripts/              # DEV TOOLS
├── tests/                # TESTING
├── ui/                   # TRADING
└── utils/                # MIXED
```

### Proposed Structure (Separated)

```
project/
├── src/
│   ├── trading/                    # 🏦 PRODUCTION TRADING CODE
│   │   ├── __init__.py
│   │   ├── algorithms/             # Trading algorithms
│   │   ├── execution/              # Order execution, fills, slippage
│   │   ├── agents/                 # LLM trading agents (from llm/agents/)
│   │   ├── analytics/              # Greeks, IV, pricing
│   │   ├── backtesting/            # Walk-forward, Monte Carlo
│   │   ├── compliance/             # Audit, FINRA, anti-manipulation
│   │   ├── data/                   # Custom data loaders
│   │   ├── indicators/             # Technical indicators
│   │   ├── infrastructure/         # Redis, pubsub, timeseries
│   │   ├── llm/                    # LLM core (sentiment, news, ensemble)
│   │   ├── mcp/                    # MCP servers
│   │   ├── models/                 # ML models, risk models
│   │   ├── scanners/               # Market scanners
│   │   ├── api/                    # REST/WebSocket APIs
│   │   ├── ui/                     # Trading dashboard
│   │   └── prompts/                # LLM prompt templates
│   │
│   └── claude_dev/                 # 🔧 DEVELOPMENT TOOLS (not deployed)
│       ├── __init__.py
│       ├── hooks/                  # Claude Code hooks (from .claude/hooks/)
│       ├── commands/               # Slash commands (from .claude/commands/)
│       ├── templates/              # Doc templates (from .claude/templates/)
│       ├── agents/                 # Dev agent configs (from .claude/agents/)
│       ├── evaluation/             # Eval frameworks (from evaluation/)
│       ├── scripts/                # Dev scripts (from scripts/)
│       ├── observability/          # Monitoring (from observability/)
│       └── utils/                  # Dev utilities (from utils/)
│
├── tests/                          # 🧪 ALL TESTS
│   ├── trading/                    # Trading tests
│   │   ├── unit/
│   │   └── integration/
│   └── claude_dev/                 # Dev tool tests
│       └── hooks/
│
├── docs/                           # 📚 DOCUMENTATION
│   ├── trading/                    # Trading docs
│   │   ├── architecture/
│   │   ├── strategies/
│   │   └── quantconnect/
│   ├── development/                # Dev workflow docs
│   │   ├── ric-workflow/
│   │   └── autonomous-agents/
│   └── research/                   # Research docs
│
├── config/                         # ⚙️ CONFIGURATION
│   ├── trading/                    # Trading config
│   │   ├── settings.json
│   │   └── watchdog.json
│   └── claude/                     # Claude Code config
│       ├── settings.json
│       └── registry.json
│
├── .claude/                        # Claude Code runtime (minimal)
│   ├── settings.json → config/claude/settings.json (symlink)
│   ├── state/                      # Runtime state
│   └── traces/                     # Debug traces
│
├── docker/                         # 🐳 DEPLOYMENT
│   ├── trading/
│   │   └── Dockerfile
│   └── development/
│       └── Dockerfile.dev
│
├── logs/                           # 📝 OUTPUT (gitignored)
├── results/                        # 📊 OUTPUT (gitignored)
├── .backups/                       # 💾 BACKUPS (gitignored)
│
├── pyproject.toml                  # Project config with workspaces
├── requirements.txt                # Production deps
├── requirements-dev.txt            # Dev deps (includes -r requirements.txt)
└── README.md
```

---

## Key Principles

### 1. `src/` Layout (Industry Standard)

From [The Hitchhiker's Guide to Python](https://docs.python-guide.org/writing/structure/):
> "Using a `src/` layout clearly separates your package from auxiliary files (tests, docs, scripts) and prevents accidental imports."

Benefits:
- Clear separation of "the product" from supporting files
- Prevents importing test code accidentally
- Standard pattern recognized by all Python tools

### 2. Separate Dependencies

From [Real Python](https://realpython.com/lessons/separating-development-and-production-dependencies/):
> "Your development environment needs additional dependencies that production doesn't need."

**requirements.txt** (production):
```
# Trading dependencies only
pandas>=2.0
numpy>=1.24
quantconnect
ccxt
redis
fastapi
```

**requirements-dev.txt** (development):
```
# Include production deps
-r requirements.txt

# Plus dev tools
pytest
black
ruff
mypy
anthropic  # For Claude Code
```

### 3. Trading Bot Best Practices

From [Alpaca's Guide](https://alpaca.markets/learn/algorithmic-trading-bot-7-steps):
> "Keep the code that talks to a broker small and audited. Enforce strict separation between 'decide' and 'do.'"

Structure reflects this:
- `execution/` - "Do" (order execution, fills)
- `agents/` + `llm/` - "Decide" (analysis, signals)
- `compliance/` - "Guard" (risk checks, audit)

---

## Migration Plan

### Phase 1: Create Directory Structure

```bash
# Create src/ layout
mkdir -p src/trading src/claude_dev
mkdir -p tests/trading/unit tests/trading/integration tests/claude_dev
mkdir -p docs/trading docs/development
mkdir -p config/trading config/claude
mkdir -p docker/trading docker/development

# Create __init__.py files
touch src/__init__.py
touch src/trading/__init__.py
touch src/claude_dev/__init__.py
```

### Phase 2: Move Trading Code

```bash
cd /home/dshooter/projects/Claude_code_Quantconnect_trading_bot

# Core trading
mv algorithms src/trading/
mv execution src/trading/
mv compliance src/trading/

# Analysis
mv analytics src/trading/
mv backtesting src/trading/
mv indicators src/trading/
mv scanners src/trading/

# Infrastructure
mv infrastructure src/trading/
mv mcp src/trading/
mv api src/trading/
mv data src/trading/
mv ui src/trading/

# LLM Trading Agents
mv llm src/trading/
mv prompts src/trading/
mv reasoning_chains src/trading/
mv models src/trading/
```

### Phase 3: Move Development Tools

```bash
# Claude Code tools
mv .claude/hooks src/claude_dev/
mv .claude/commands src/claude_dev/
mv .claude/templates src/claude_dev/
mv .claude/agents src/claude_dev/

# Dev scripts and evaluation
mv scripts src/claude_dev/
mv evaluation src/claude_dev/
mv observability src/claude_dev/
mv utils src/claude_dev/
```

### Phase 4: Reorganize Tests

```bash
# Move tests to match new structure
mv tests/analytics tests/trading/unit/
mv tests/backtesting tests/trading/unit/
mv tests/compliance tests/trading/unit/
mv tests/infrastructure tests/trading/unit/
mv tests/mcp tests/trading/unit/

mv integration_tests tests/trading/integration/
mv tests/hooks tests/claude_dev/
mv tests/observability tests/claude_dev/
```

### Phase 5: Update Imports

All imports need updating. Example transformations:

```python
# OLD
from algorithms.basic_buy_hold import BasicBuyHold
from llm.agents.technical_analyst import TechnicalAnalyst
from execution.smart_execution import SmartExecution

# NEW
from trading.algorithms.basic_buy_hold import BasicBuyHold
from trading.agents.technical_analyst import TechnicalAnalyst
from trading.execution.smart_execution import SmartExecution
```

**Automated approach:**
```bash
# Find and replace imports
find src/ -name "*.py" -exec sed -i 's/from algorithms\./from trading.algorithms./g' {} \;
find src/ -name "*.py" -exec sed -i 's/from execution\./from trading.execution./g' {} \;
find src/ -name "*.py" -exec sed -i 's/from llm\./from trading.llm./g' {} \;
# ... etc for all modules
```

### Phase 6: Update Claude Code Paths

Update `.claude/settings.json` hook paths:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "command": "python3 src/claude_dev/hooks/core/ric.py",
        ...
      }
    ]
  }
}
```

Update `config/claude/registry.json` similarly.

### Phase 7: Create pyproject.toml

```toml
[project]
name = "quantconnect-trading-bot"
version = "1.0.0"
description = "Algorithmic trading bot with LLM agents"
requires-python = ">=3.10"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"

[tool.ruff]
src = ["src"]
exclude = [".claude/state", "logs", "results"]

[tool.mypy]
packages = ["trading", "claude_dev"]
```

### Phase 8: Cleanup

```bash
# Remove deprecated
rm -rf agents_deprecated

# Remove empty directories
find . -type d -empty -delete

# Update .gitignore
cat >> .gitignore << 'EOF'
# Output directories
logs/
results/
.backups/

# Claude Code runtime
.claude/state/
.claude/traces/

# Build artifacts
*.egg-info/
dist/
build/
EOF
```

---

## Directory Ownership

### `src/trading/` - Production Code
- **Owner:** Trading/Quant Team
- **Deployed to:** Production servers, QuantConnect
- **Changes require:** Code review, backtesting
- **Dependencies:** Production only (requirements.txt)

### `src/claude_dev/` - Development Tools
- **Owner:** Development/DevOps
- **Deployed to:** Development machines only
- **Changes require:** Testing hooks work
- **Dependencies:** Dev tools (requirements-dev.txt)

---

## Import Guidelines

### From Trading Code
```python
# Internal trading imports
from trading.algorithms import BasicBuyHold
from trading.execution import SmartExecution
from trading.agents import TechnicalAnalyst

# DO NOT import from claude_dev in trading code!
# ❌ from claude_dev.hooks import something
```

### From Development Tools
```python
# Dev tools CAN import from trading (for testing/evaluation)
from trading.algorithms import BasicBuyHold
from claude_dev.evaluation import run_backtest
```

---

## Verification Checklist

After migration, verify:

- [ ] `python -c "from trading import algorithms"` works
- [ ] `python -c "from trading.agents import TechnicalAnalyst"` works
- [ ] `python -c "from claude_dev.hooks.core import ric"` works
- [ ] `pytest tests/trading/` passes
- [ ] `pytest tests/claude_dev/` passes
- [ ] Claude Code hooks still function
- [ ] RIC loop works: `python src/claude_dev/hooks/core/ric.py status`
- [ ] No circular imports
- [ ] Docker build succeeds

---

## Rollback Plan

If migration fails:

```bash
# Git reset to before migration
git log --oneline -10
git reset --hard <commit-before-migration>
```

---

## Benefits After Migration

1. **Clear Boundaries:** Trading code is isolated in `src/trading/`
2. **Deployable:** Can deploy `src/trading/` without dev tools
3. **Testable:** Tests mirror src structure
4. **Maintainable:** Clear ownership of each section
5. **Industry Standard:** Follows Python packaging best practices
6. **Dependency Separation:** Production vs dev dependencies clear

---

## Timeline Estimate

| Phase | Description | Effort |
|-------|-------------|--------|
| 1 | Create directory structure | 10 min |
| 2 | Move trading code | 30 min |
| 3 | Move dev tools | 20 min |
| 4 | Reorganize tests | 30 min |
| 5 | Update imports | 2-4 hours |
| 6 | Update Claude paths | 30 min |
| 7 | Create pyproject.toml | 15 min |
| 8 | Cleanup & verify | 1 hour |

**Total:** ~5-6 hours of focused work

---

## Quick Start

To execute this plan:

```
Read .claude/SEPARATION_PLAN.md and execute Phase 1
```

Or execute all phases:
```
Execute the full separation plan from .claude/SEPARATION_PLAN.md
```

---

## Notes

1. **Do REFACTOR_PLAN.md first** - P0 fixes should be done before this
2. **Commit after each phase** - Easy rollback
3. **Test after Phase 5** - Import changes are error-prone
4. **Update CI/CD** - Pipeline paths will need updating
