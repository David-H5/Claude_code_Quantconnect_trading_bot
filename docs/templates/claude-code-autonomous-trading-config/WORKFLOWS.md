# Development Workflows

## Default Workflow: Explore → Plan → Code → Test → Commit

For any task, follow this sequence:

### 1. Explore
- Read relevant existing code
- Check related tests
- Review documentation
- Identify dependencies

### 2. Plan
- Write implementation approach in `plans/TASK-{id}.md`
- List files to modify
- Identify potential risks
- Define success criteria

### 3. Code
- Implement changes incrementally
- Run tests after each significant change
- Keep commits atomic

### 4. Test
- Run `make test` - all must pass
- Run `make lint` - all must pass
- For algorithms: `lean backtest` with minimum 100 trades

### 5. Commit
- Clear, conventional commit message
- Reference task/issue ID
- PR if touching production code

## Task-Specific Workflows

### New Trading Strategy

```
1. Create algorithm in algorithms/development/
2. Write comprehensive backtest scenarios
3. Run backtests across multiple time periods
4. Document results in backtests/results/
5. Paper trade for 2 weeks minimum
6. Code review required before production
```

### Bug Fix

```
1. Reproduce bug with failing test
2. Fix code
3. Verify test passes
4. Check for related issues
5. Update documentation if behavior changed
```

### Refactoring

```
1. Ensure test coverage > 80% for affected code
2. Make changes incrementally
3. Run full test suite after each step
4. No functional changes - tests should not change
```

### Infrastructure Change

```
1. Test in docker-sandbox first
2. Document rollback procedure
3. Get explicit approval for production changes
4. Deploy during low-activity periods
```

## Autonomous Session Protocol

For extended autonomous development:

### Pre-Session
1. Define clear scope in task document
2. Set time/cost limits
3. Enable sandbox mode
4. Configure hooks for safety

### During Session
- Checkpoint progress every 30 minutes to `SESSION_NOTES.md`
- Commit working code frequently
- Log blockers and decisions
- Pause for confirmation on critical paths

### Post-Session
- Summarize changes in PR description
- Run full test suite
- Document any technical debt created
- Update CLAUDE.md if new patterns established

## Git Workflow

- Branch naming: `feature/`, `fix/`, `refactor/`, `docs/`
- Commit format: `type(scope): message`
- Always rebase on main before PR
- Squash merge for features

## When to STOP and Ask

- Any production deployment
- Changes to risk limits
- New broker integrations
- Database schema changes
- Security-related modifications
- Unclear requirements
- Conflicting constraints
