# Git Workflow for Multi-Agent Development

This document outlines git best practices for projects with multiple autonomous Claude agents.

## Current Issue: Large File in History

The `llm/prompts/registry.json` file (currently 56MB, up to 106MB in history) exceeds GitHub's 100MB limit.

### Resolution Options

**Option 1: Clean History with Orphan Branch (Recommended)**
```bash
# Create orphan branch with current state only
git checkout --orphan clean-main
git add -A
git commit -m "Initial commit: Clean state from December 2025"
git branch -D main
git branch -m main
git push origin main --force
```
- Pros: Clean slate, no large file history, simple
- Cons: Loses commit history (can preserve in archive branch)

**Option 2: Git LFS Migration**
```bash
git lfs install
git lfs track "llm/prompts/registry.json"
git lfs migrate import --include="llm/prompts/registry.json"
git push origin main --force
```
- Pros: Preserves history, proper large file handling
- Cons: Requires Git LFS setup, GitHub LFS storage limits

**Option 3: Split Registry File**
Split the 116k line JSON into smaller files (<10MB each).
- Pros: No special tooling needed
- Cons: Requires code changes to load multiple files

---

## Recommended Git Structure for Multi-Agent Projects

### Branch Strategy

```
main                    # Production-ready code (protected)
├── develop             # Integration branch for agents
├── agent/overnight     # Overnight agent work
├── agent/refactor      # Refactoring agent work
├── agent/research      # Research agent work
└── feature/*           # Human feature branches
```

### Branch Rules

| Branch | Merge From | Merge To | Protection |
|--------|------------|----------|------------|
| main | develop only | - | Required reviews, tests pass |
| develop | agent/*, feature/* | main | Tests must pass |
| agent/* | develop | develop | Auto-tests only |
| feature/* | develop | develop | Human review |

### Commit Conventions

```
<type>(<scope>): <description>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- refactor: Code refactoring
- test: Adding tests
- chore: Maintenance

Examples:
feat(execution): Add two-part spread strategy
fix(circuit-breaker): Correct daily loss calculation
docs(claude): Restructure CLAUDE.md for clarity
```

### Agent Commit Signature

All agent commits should include:
```
🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

---

## Multi-Agent Coordination

### File Locking Protocol

For files that multiple agents might modify simultaneously:

```python
# Use fcntl file locking (implemented in utils/overnight_state.py)
import fcntl

with open(state_file, 'r+') as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
    # Read/modify/write
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release
```

### State Files (Do Not Track)

These files change frequently and should be in `.gitignore`:
```
.claude/state/ric.json
bot_positions.json
circuit_breaker_log.json
*.log
```

### Protected Files (Require Human Review)

```
algorithms/*.py          # Trading algorithms
config/settings.json     # Production config
.claude/settings.json    # Claude settings
```

### Checkpoint Protocol

Before major operations, agents should:
```bash
git add -A
git commit -m "checkpoint: <description>"
git tag checkpoint-$(date +%Y%m%d-%H%M)
```

---

## Recommended Hooks

### Pre-commit (Fast, <30s)

```yaml
# .pre-commit-config.yaml priorities
hooks:
  - trailing-whitespace    # Instant
  - end-of-file-fixer      # Instant
  - check-json             # Fast
  - check-yaml             # Fast
  - ruff                   # Fast (200x faster than flake8)
```

### Pre-push (Thorough, <5min)

```yaml
hooks:
  - pytest tests/ -v -m unit
  - mypy algorithms/ models/ execution/
```

### Excluded from Hooks

Large or frequently-changing files:
```yaml
exclude: |
  llm/prompts/registry\.json|
  \.hypothesis/|
  reasoning_chains/
```

---

## Conflict Resolution

### Auto-resolvable Conflicts

For JSON state files, use "ours" strategy:
```bash
git config merge.ours.driver true
echo "*.json merge=ours" >> .gitattributes
```

### Manual Resolution Required

- Python code conflicts → Human review
- Configuration changes → Human review
- Algorithm changes → Human review + backtest

---

## Recovery Procedures

### Agent Session Crashed

```bash
# Check last good state
git reflog
git log --oneline -10

# Recover to last checkpoint
git reset --hard checkpoint-YYYYMMDD-HHMM

# Or recover specific file
git checkout HEAD~5 -- path/to/file
```

### Merge Conflict in Overnight Session

```bash
# Stash agent changes
git stash

# Get clean develop
git checkout develop
git pull origin develop

# Create new agent branch
git checkout -b agent/retry-$(date +%Y%m%d)
git stash pop
```

---

## Recommended .gitignore Additions

```gitignore
# Agent artifacts
.hypothesis/
reasoning_chains/
*.log
*_backup.json

# State files (change frequently)
.claude/state/
bot_positions.json
circuit_breaker_log.json

# Large generated files
llm/prompts/registry.json.backup
```

---

## Monitoring Agent Activity

### View Agent Commits

```bash
# All agent commits
git log --author="Claude" --oneline

# Today's agent activity
git log --since="midnight" --oneline

# Changes by agents
git shortlog -sn --author="Claude"
```

### Audit Trail

Track agent decisions in `logs/agent_decisions.jsonl`:
```json
{"timestamp": "2025-12-06T14:00:00Z", "agent": "overnight", "action": "commit", "files": 5}
```

---

## Testing Best Practices for Agents

### Prevent Test Pollution

Tests must not write to production files. Use these patterns:

```python
# GOOD: Disable logging in tests
manager = create_bot_position_manager(algorithm, enable_logging=False)

# GOOD: Use temp directories
@pytest.fixture
def components(self, algorithm, tmp_path):
    return {
        "manager": create_manager(storage_path=tmp_path),
    }

# BAD: Uses default paths that write to repo
manager = BotPositionManager(algorithm=None)  # Writes to bot_positions.json!
```

### Test Isolation Checklist

| Check | How to Verify |
|-------|---------------|
| No file pollution | `git status` clean after tests |
| Temp paths used | Search for `tmp_path` fixture usage |
| Logging disabled | Search for `enable_logging=False` |
| Mocks for external | `@patch` decorators present |

---

## Agent Safety Protocols

### Pre-Commit Validation

Agents should validate changes before committing:

```python
# Run validation checks
python scripts/check_layer_violations.py --strict
python scripts/qa_validator.py --check debug --check integrity
python scripts/algorithm_validator.py algorithms/*.py
```

### Skip Hooks During Overnight Sessions

When running overnight with background processes:
```bash
# Skip hooks that might conflict with running processes
SKIP=qa-validator git commit -m "overnight: checkpoint"

# Or skip all for emergency commits
git commit --no-verify -m "emergency: fix critical bug"
```

### Circuit Breaker Integration

Agents must respect trading halts:

```python
from models.circuit_breaker import get_circuit_breaker

breaker = get_circuit_breaker()
if not breaker.can_trade():
    logger.warning("Trading halted - skipping autonomous actions")
    return
```

---

## Multi-Agent Communication

### Handoff Protocol

When one agent hands off to another:

1. **Commit checkpoint**: `git commit -m "handoff: <agent> → <next-agent>"`
2. **Update state file**: Write to `.claude/state/handoff.json`
3. **Log context**: Include reasoning in commit message

```json
// .claude/state/handoff.json
{
  "from_agent": "overnight",
  "to_agent": "morning",
  "timestamp": "2025-12-06T06:00:00Z",
  "context": {
    "completed": ["backtest SPY strategy", "updated indicators"],
    "pending": ["review iron condor performance"],
    "warnings": ["circuit breaker triggered at 3am"]
  }
}
```

### Shared Resource Access

| Resource | Lock Type | Location |
|----------|-----------|----------|
| bot_positions.json | fcntl LOCK_EX | runtime |
| registry.json | git merge | commit time |
| config/settings.json | human approval | manual |

---

## Performance Optimization

### Reduce Hook Execution Time

```yaml
# Exclude large files from ALL hooks
exclude: |
  llm/prompts/registry\.json|
  \.hypothesis/|
  reasoning_chains/|
  \.backups/
```

### Parallel Validation

```bash
# Run validations in parallel
python scripts/check_layer_violations.py &
python scripts/qa_validator.py &
wait  # Wait for all to complete
```

### Memory-Efficient Git Operations

```bash
# Avoid loading large files in memory
git diff --stat HEAD  # May crash on large repos
git status --short    # Safer alternative

# Clean up if git becomes slow
git gc --prune=now
git repack -a -d
```

---

## Troubleshooting

### Bus Error During Git Operations

```bash
# Remove stale lock files
rm -f .git/index.lock

# Run garbage collection
git gc --prune=now

# If persists, re-clone or use orphan branch
```

### Pre-commit Hook Failures

| Error | Solution |
|-------|----------|
| `ruff` failures | Fix or `SKIP=ruff git commit` |
| `mypy` type errors | Fix or `SKIP=mypy git commit` |
| `qa-validator` modified files | `SKIP=qa-validator git commit` |
| Large file detected | Add to `.gitignore` or use LFS |

### Agent Conflict with Human Edits

```bash
# Save agent work
git stash

# Apply human changes
git pull origin main

# Reapply agent work
git stash pop

# Resolve conflicts manually, then:
git add -A
git commit -m "merge: resolved human/agent conflicts"
```
