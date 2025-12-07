# Claude Code Instructions - QuantConnect Trading Bot

## Overview

Semi-autonomous options trading bot with LLM-powered analysis.

- **Platform**: QuantConnect/LEAN with Charles Schwab brokerage
- **Language**: Python 3.10+
- **Testing**: pytest (70% minimum coverage)
- **UI**: PySide6 trading dashboard

## Critical: TAP Protocol (Thorough Action Protocol)

**Every change MUST follow: LOG -> CHECK -> TEST -> INTEGRATE -> VERIFY -> LOG**

| Step | Action | Required Output |
|------|--------|-----------------|
| 1. LOG START | Log what you're about to do | `[TAP] Starting: <action>` |
| 2. CHECK | Verify prerequisites exist | List files/functions that must exist |
| 3. IMPLEMENT | Make the change | Show the actual code change |
| 4. TEST | Run relevant tests | `pytest <test_file> -v` output |
| 5. INTEGRATE | Wire into existing code | Show import/usage added |
| 6. VERIFY | Confirm integration works | `python -c "from X import Y"` |
| 7. LOG END | Document what was done | `[TAP] Complete: <result>` |

**Never skip tests. Never create without integrating. Never assume - verify.**

## Essential Commands

### RIC Loop (Complex Tasks)

See [RIC Loop Workflow](#ric-loop-workflow-complex-tasks) below for full details.

| Command | Description |
|---------|-------------|
| `/ric-start <task>` | Start RIC Loop session |
| `/ric-research <topic>` | Begin Phase 0 research |
| `/ric-converge` | Check convergence and decide next step |

### Multi-Agent Orchestration

| Command | Description |
|---------|-------------|
| `/agents auto <task>` | Auto-route to best agents |
| `/agent-quick <task>` | Single fast agent |
| `/agent-swarm <topic>` | 8-agent parallel exploration |
| `/agent-consensus <decision>` | Multi-agent voting |

### Development

| Command | Description |
|---------|-------------|
| `pytest tests/ -v` | Run all tests |
| `ruff check .` | Lint code |
| `mypy .` | Type check |
| `/save-research <type> "Title"` | Save research document |

## Project Structure

```
algorithms/     # QuantConnect trading algorithms
llm/agents/     # LLM trading agents (primary)
execution/      # Order execution (profit-taking, smart-exec)
scanners/       # Market scanners (options, movement)
models/         # Risk models, circuit breaker
tests/          # Unit and integration tests
.claude/hooks/  # Claude Code hooks and orchestration
```

See: @docs/ARCHITECTURE.md for full structure

## RIC Loop Workflow (Complex Tasks)

For multi-file changes, features, and refactoring use the RIC Loop:

```
P0 -> P1 -> P2 -> P3 -> P4 -> Loop/Exit
RESEARCH  PLAN  BUILD  VERIFY  REFLECT
```

**Rules:**
- Minimum 3 iterations before exit allowed
- All P0/P1/P2 insights must be resolved before exit
- Persist research to `docs/research/` every 3 searches

See: @.claude/RIC_CONTEXT.md for quick reference

## Development Workflow

### For Complex Tasks (multi-file, features, refactoring)

1. Start RIC Loop: `/ric-start <task>`
2. Research -> Plan -> Build -> Verify -> Reflect
3. Loop until all insights resolved
4. Exit with finalization checklist complete

### For Simple Tasks (single file, bug fixes)

1. Make change
2. Run relevant tests
3. Verify imports work
4. Commit with clear message

## Safety Rules

- **NEVER** deploy untested code to live trading
- **ALWAYS** verify circuit breaker integration
- **REQUIRE** human approval for production trades
- **TEST** in paper trading before live

See: @docs/SAFETY_GUIDE.md for complete safety checklist

## Code Quality

Use deterministic tools (not Claude instructions):

```bash
ruff format .     # Formatting
ruff check . --fix  # Linting
mypy .            # Type checking
```

## Common Pitfalls

1. Don't modify live algorithm without backtest
2. Don't skip paper trading validation
3. Don't commit without running tests
4. Don't ignore circuit breaker warnings
5. Don't hold research only in context (persist to files)

## Key References

| Topic | File |
|-------|------|
| RIC Loop | @.claude/RIC_CONTEXT.md |
| Architecture | @docs/ARCHITECTURE.md |
| Trading Patterns | @docs/OPTIONS_PATTERNS.md |
| Agent System | @docs/AGENT_ORCHESTRATION.md |
| Agent Personas | @docs/AGENT_PERSONAS.md |
| Safety | @docs/SAFETY_GUIDE.md |
| QuantConnect | @docs/QUANTCONNECT_GUIDE.md |
| Research Index | @docs/research/README.md |
| Development | @docs/development/BEST_PRACTICES.md |

## Quick Reference: Charles Schwab

- **ONE algorithm per account** (second deploys stop first)
- Greeks use IV, **no warmup required**
- Use `ComboLimitOrder()` for multi-leg (not `ComboLegLimitOrder`)
- OAuth re-auth required ~weekly

## Quick Reference: Testing

```bash
# All tests
pytest tests/ -v

# With coverage
pytest --cov=. --cov-fail-under=70

# Specific test
pytest tests/test_file.py -v -k test_name
```

## Research Documentation

When conducting online research:

1. Create doc in `docs/research/` with timestamped sources
2. Persist findings immediately (context can compact)
3. Update `docs/research/README.md` index

Use: `/save-research upgrade "Title"` for proper templates

## External Resources

- [QuantConnect Docs](https://www.quantconnect.com/docs)
- [LEAN GitHub](https://github.com/QuantConnect/Lean)
- [Charles Schwab API](https://developer.schwab.com/)
