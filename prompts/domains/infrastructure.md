# Infrastructure Domain Context

You are working on **CI/CD, deployment, Docker, or hooks** code.

## GitHub Actions Workflows

| Workflow | Purpose | Trigger |
|----------|---------|---------|
| `autonomous-testing.yml` | Full test pipeline | Push to develop/feature |
| `paper-trading.yml` | Paper trading deployment | Manual |
| `branch-protection.yml` | Enforce branching rules | PRs |
| `deploy.yml` | Deploy to QuantConnect | Push to main |

## Branch Strategy

| Branch | Purpose | Protection |
|--------|---------|------------|
| `main` | Production | Requires PR from develop |
| `develop` | Integration | Requires PR approval |
| `feature/*` | New features | None |
| `hotfix/*` | Urgent fixes | Can PR to main |

## Docker Configuration

```bash
# Build sandbox
docker build -t trading-sandbox .

# Run with security restrictions
docker run -it \
    --cap-drop ALL \
    --memory=2g \
    --network=none \
    trading-sandbox
```

## Claude Code Hooks

Hooks are configured in `.claude/settings.json`:

| Hook | Trigger | Purpose |
|------|---------|---------|
| `protect_files.py` | PreToolUse | Block sensitive files |
| `ric_hooks.py` | PreToolUse + UserPromptSubmit | RIC enforcement |
| `validate_algorithm.py` | PostToolUse | Validate algo changes |
| `session_stop.py` | Stop | Session cleanup |
| `pre_compact.py` | PreCompact | Transcript backup |

## Compute Nodes (QuantConnect)

| Type | Model | Purpose |
|------|-------|---------|
| Backtesting | B8-16 | Options data processing |
| Research | R8-16 | LLM ensemble |
| Live Trading | L2-4 | Real-time trading |

## Before Committing

- [ ] Rollback plan documented
- [ ] Staging test completed
- [ ] Security implications reviewed
- [ ] No secrets in code
