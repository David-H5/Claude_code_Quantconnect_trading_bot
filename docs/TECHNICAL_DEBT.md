# Technical Debt Report

*Generated: 2025-12-05 20:06*

## Summary

| Priority | Count | Description |
|----------|------:|-------------|
| **P0** | 0 | Critical - blocks functionality |
| **P1** | 2 | Important - should fix soon |
| **P2** | 4 | Polish - nice to have |
| **Total** | **6** | |

## P1 Items (2)

### `scripts/scan_todos.py`

- [ ] **Line 95** [TODO]: , FIXME, XXX, HACK, BUG
- [ ] **Line 148** [BUG]: and FIXME default to higher priority

## P2 Items (4)

### `evaluation/continuous_monitoring.py`

- [ ] **Line 417** [TODO]: Implement email/Slack/webhook notifications

### `evaluation/test_data_validation.py`

- [ ] **Line 338** [TODO]: Implement redundancy detection

### `execution/arbitrage_executor.py`

- [ ] **Line 762** [TODO]: Calculate IV edge

### `tests/test_example.py`

- [ ] **Line 49** [TODO]: Add tests for your algorithms, indicators, and utilities

## Addressing Technical Debt

### Priority Guidelines

- **P0**: Address immediately - these block functionality or pose security risks
- **P1**: Plan for current or next sprint - bugs and important improvements
- **P2**: Add to backlog - address during refactoring sessions

### Workflow

1. Pick an item from P0 (if any), otherwise P1
2. Create a branch: `git checkout -b fix/todo-description`
3. Fix the issue and remove the TODO comment
4. Run tests: `pytest tests/ -v`
5. Commit with message: `fix: resolve TODO - description`

### Regenerate Report

```bash
python scripts/scan_todos.py
```
