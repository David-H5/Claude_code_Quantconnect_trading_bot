# General Domain Context

You are working on a task that doesn't fall into a specific domain category.

## Project Overview

This is a **QuantConnect trading bot** project with:

- Python 3.10+ codebase
- Semi-autonomous development with Claude Code
- LLM-powered analysis ensemble
- Options trading focus (Charles Schwab brokerage)

## Code Quality Standards

- **Type hints** on all methods (Python 3.8+ compatible: `List[X]` not `list[X]`)
- **Google-style docstrings**
- **Max 100 characters** per line
- **ruff** for linting
- **black** for formatting
- **mypy** for type checking

## Quick Commands

```bash
# Lint
ruff check .

# Format
black .

# Type check
mypy --config-file mypy.ini

# Tests
pytest tests/ -v
```

## Safety First

- Never deploy untested code
- Always use circuit breaker for trading
- Backup before modifying algorithm files
- No secrets in code

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `algorithms/` | Trading algorithms |
| `llm/` | LLM integration |
| `execution/` | Order execution |
| `models/` | Risk models |
| `tests/` | Test suite |
| `evaluation/` | Evaluation frameworks |

## Before Committing

- [ ] Tests pass
- [ ] Linting clean
- [ ] No secrets committed
- [ ] Change is minimal and focused
