# Sync Claude Code Registry

Synchronize the Claude Code registry with discovered hooks, scripts, and commands.

## What This Does

1. **Discovers** new hooks in `.claude/hooks/`
2. **Discovers** new scripts in `scripts/`
3. **Discovers** new commands in `.claude/commands/`
4. **Updates** `.claude/registry.json` with new components
5. **Syncs** `.claude/settings.json` to ensure hooks are configured
6. **Validates** all components work correctly
7. **Enforces** mandatory hooks are present

## Run Sync

```bash
# Full sync (discovers, configures, validates)
python scripts/sync_claude_registry.py

# Check only (no changes)
python scripts/sync_claude_registry.py --check

# Validate components only
python scripts/sync_claude_registry.py --validate

# Enforce mandatory hooks only
python scripts/sync_claude_registry.py --enforce
```

## Mandatory Hooks

The following hooks are **required** and will be enforced:

| Hook | Purpose |
|------|---------|
| `protect_files` | Prevents access to sensitive files |
| `ric` | RIC Loop v5.1 - Unified enforcement and suggestions |
| `validate_algorithm` | Validates algorithm files |
| `validate_research` | Validates research document naming |
| `document_research` | Reminds to document after web searches |

## Mandatory Workflows

| Workflow | When Required |
|----------|---------------|
| RIC Loop | Multi-file changes, research tasks, architecture decisions |
| Documentation | Web searches, research findings |
| Algorithm Safety | Algorithm modifications, deployment |

## Registry Files

| File | Purpose |
|------|---------|
| `.claude/registry.json` | Master registry of all components |
| `.claude/settings.json` | Hook configurations |
| `docs/development/DOCUMENTATION_HOOKS.md` | Hook documentation |

## When to Run

Run sync when:
- Adding a new hook to `.claude/hooks/`
- Adding a new script to `scripts/`
- Adding a new slash command to `.claude/commands/`
- After pulling changes that may include new components
- When hooks aren't working as expected

## Output

```
✓ [enforce] hook:protect_files: Mandatory hook configured
✓ [validate] hook:validate_research: Syntax valid
+ [discover] script:new_script: New script discovered
```

- `✓` = OK
- `+` = Added
- `⚠` = Warning
- `✗` = Error
