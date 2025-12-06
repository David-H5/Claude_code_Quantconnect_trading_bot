# Overnight System Refactoring Plan v2.0

**Created**: 2025-12-06
**Based on**: Web research + OVERNIGHT_SYSTEM_ANALYSIS.md analysis
**Priority**: HIGH - Infrastructure improvement
**Estimated Effort**: 5-7 days

---

## Research Summary

### Sources Consulted

1. [Claude Code Best Practices - Anthropic](https://www.anthropic.com/engineering/claude-code-best-practices)
2. [Effective Harnesses for Long-Running Agents - Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
3. [Durable AI Loops - Restate](https://www.restate.dev/blog/durable-ai-loops-fault-tolerance-across-frameworks-and-without-handcuffs)
4. [Persistence in LangGraph](https://medium.com/@iambeingferoz/persistence-in-langgraph-building-ai-agents-with-memory-fault-tolerance-and-human-in-the-loop-d07977980931)
5. [Microsoft Durable Task Extension](https://techcommunity.microsoft.com/blog/appsonazureblog/bulletproof-agents-with-the-durable-task-extension-for-microsoft-agent-framework/4467122)

### Key Patterns Identified

1. **Two-Agent System** (Anthropic): Initializer + Coding agent pattern
2. **Durable Execution** (Restate/Microsoft): Virtual objects with identity + state
3. **Checkpointing** (LangGraph): Resume from last checkpoint on failure
4. **Shift-Handoff Documentation** (Anthropic): Clear artifacts for next session

---

## Current State Assessment

### Already Implemented (OVERNIGHT-002)
- [x] utils/overnight_state.py - Unified state with file locking
- [x] utils/overnight_config.py - Centralized configuration
- [x] utils/progress_parser.py - Unified progress parsing
- [x] Fixed exception suppression in hook_utils.py
- [x] Fixed git --no-verify bypass in session_stop.py
- [x] config/overnight.yaml - Configuration file

### Pending from OVERNIGHT-002
- [ ] Migrate existing code to use unified state manager
- [ ] Add tests for state/config/parser
- [ ] Replace duplicate parsing code
- [ ] Add Claude CLI availability check
- [ ] Create health check API

---

## Refactoring Plan v2

### Phase 1: Durable Execution Layer (P0 - Days 1-2)

Based on Restate/Microsoft patterns, implement durable execution:

```python
# New file: utils/durable_session.py
class DurableSession:
    """Durable session with automatic persistence and recovery."""

    session_id: str
    state: OvernightState
    checkpoints: list[Checkpoint]

    def checkpoint(self, label: str) -> Checkpoint
    def restore(self, checkpoint_id: str) -> bool
    def execute_with_retry(self, action: Callable) -> Any
```

**Tasks:**
- [ ] Create utils/durable_session.py with checkpoint support
- [ ] Implement automatic state persistence on each action
- [ ] Add transaction-like commit/rollback for multi-step operations
- [ ] Integrate with existing OvernightStateManager

### Phase 2: Two-Agent Architecture (P0 - Days 2-3)

Based on Anthropic's research, implement initializer/coding agent pattern:

**Initializer Agent** (runs once per project):
- Set up environment (init.sh)
- Create progress tracking (claude-progress.txt)
- Initialize git baseline
- Create feature registry

**Session Agent** (runs each session):
- Review progress file and git log
- Identify next priority work
- Make incremental progress
- Create handoff documentation

**Tasks:**
- [ ] Create scripts/initializer_agent.py
- [ ] Create scripts/session_agent.py
- [ ] Define feature registry format (JSON)
- [ ] Implement session handoff protocol
- [ ] Update run_overnight.sh to use two-agent pattern

### Phase 3: Enhanced State Persistence (P1 - Days 3-4)

Based on LangGraph patterns, implement multi-layer persistence:

```
State Layers:
├── Transient (in-memory)
│   └── Current task, active decisions
├── Session (session_state.json)
│   └── Progress, blockers, recent actions
├── Persistent (git + progress.txt)
│   └── Completed work, feature status
└── Long-term (feature_registry.json)
    └── Project scope, completed features
```

**Tasks:**
- [ ] Implement StateLayer enum and manager
- [ ] Create feature_registry.json format
- [ ] Add automatic layer synchronization
- [ ] Implement conflict resolution for concurrent access

### Phase 4: Recovery Enhancement (P1 - Days 4-5)

Based on crash recovery research:

**Tasks:**
- [ ] Implement checkpoint-based recovery (git tags)
- [ ] Add rollback capability for failed features
- [ ] Create recovery decision tree
- [ ] Implement baseline testing before new work
- [ ] Add environment health verification

### Phase 5: Observability & Monitoring (P2 - Days 5-6)

**Tasks:**
- [ ] Create scripts/health_check.py HTTP server
- [ ] Implement /health and /status endpoints
- [ ] Add Prometheus-compatible metrics
- [ ] Create session timeline visualization
- [ ] Integrate with watchdog.py

### Phase 6: Integration & Testing (P2 - Days 6-7)

**Tasks:**
- [ ] Migrate all scripts to use new infrastructure
- [ ] Add comprehensive test suite
- [ ] Create integration tests for full session lifecycle
- [ ] Document new architecture
- [ ] Update CLAUDE.md with new patterns

---

## Implementation Details

### Durable Session API

```python
from utils.durable_session import DurableSession

async with DurableSession.start("session-001") as session:
    # Automatic checkpointing
    session.checkpoint("before_feature_x")

    try:
        result = await session.execute_with_retry(
            implement_feature_x,
            max_retries=3,
            backoff="exponential"
        )
        session.commit("feature_x_complete")
    except Exception:
        session.rollback("before_feature_x")
```

### Feature Registry Format

```json
{
  "project": "trading-bot",
  "version": "1.0.0",
  "features": [
    {
      "id": "overnight-refactor",
      "status": "in_progress",
      "priority": "P0",
      "started": "2025-12-06",
      "phases": [
        {"name": "state_consolidation", "complete": true},
        {"name": "durable_execution", "complete": false}
      ]
    }
  ],
  "completed_features": []
}
```

### Session Handoff Protocol

Each session ends with:
1. Progress commit to git
2. Update claude-progress.txt with:
   - What was accomplished
   - What's next
   - Any blockers
3. Feature registry update
4. Health check verification

Each session starts with:
1. Load last checkpoint
2. Review progress file
3. Verify environment health
4. Load feature registry
5. Identify next priority task

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Session recovery success rate | > 95% |
| State persistence reliability | 100% |
| Time to resume after crash | < 60s |
| Context preservation across compactions | 100% |
| Test coverage for overnight code | > 80% |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing workflows | Gradual migration with compatibility layer |
| State corruption | Atomic writes, checksums, recovery points |
| Complex rollback scenarios | Clear transaction boundaries, testing |
| Performance overhead | Lazy persistence, batched writes |

---

## Timeline

| Day | Focus | Deliverables |
|-----|-------|-------------|
| 1 | Durable execution foundation | durable_session.py, basic tests |
| 2 | Checkpoint/restore | Full checkpoint system |
| 3 | Two-agent architecture | initializer/session agents |
| 4 | Recovery enhancement | Rollback, baseline testing |
| 5 | Observability | Health API, metrics |
| 6 | Integration | Script migration |
| 7 | Testing & docs | Full test suite, documentation |

---

## References

- [Anthropic Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- docs/OVERNIGHT_SYSTEM_ANALYSIS.md (internal)
- utils/overnight_state.py (already implemented)
- utils/overnight_config.py (already implemented)
