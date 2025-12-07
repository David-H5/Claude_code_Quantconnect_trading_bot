# Parallel Upgrade Coordination System (PUCS)

**Version:** 1.1 (Consolidated)
**Date:** December 6, 2025
**Status:** Proposal
**Supersedes:** [PARALLEL_AGENT_COORDINATION.md](PARALLEL_AGENT_COORDINATION.md)

> **Note:** This document consolidates best practices from both the original
> PARALLEL_AGENT_COORDINATION.md and new research into a single comprehensive guide.

## Executive Summary

This document proposes a **Parallel Upgrade Coordination System (PUCS)** to enable multiple AI agents to work on the codebase simultaneously without conflicts. Based on analysis of 30,587 lines of code across 150+ modules and industry best practices, we identify **7 independent work streams** and propose coordination mechanisms to maximize parallelization while preventing merge conflicts.

---

## 1. Codebase Analysis Summary

### 1.1 Architecture Layers

```
Layer 4: Applications   → algorithms/, ui/, api/
Layer 3: Business Logic → execution/, evaluation/, mcp/
Layer 2: Core Analysis  → llm/, models/, scanners/, indicators/, backtesting/
Layer 1: Infrastructure → config/, observability/, utils/, compliance/
```

### 1.2 Module Statistics

| Category | Files | Lines | Independence |
|----------|-------|-------|--------------|
| Execution | 18 | ~3,000 | 95% |
| LLM/Agents | 40 | ~8,000 | 90% |
| Scanners | 5 | ~1,200 | 98% |
| Models | 22 | ~4,500 | 85% |
| Evaluation | 30+ | ~5,000 | 100% |
| Observability | 16 | ~2,500 | 100% |
| API | 10+ | ~2,000 | 90% |

### 1.3 Critical Shared Files (Require Coordination)

| File | Used By | Risk |
|------|---------|------|
| `config/__init__.py` | ALL modules | HIGH |
| `models/circuit_breaker.py` | algorithms, execution, api | HIGH |
| `models/risk_manager.py` | algorithms, execution, api | HIGH |
| `algorithms/base_options_bot.py` | all algorithms | MEDIUM |
| `llm/agents/base.py` | all agents | MEDIUM |

---

## 2. Identified Parallel Work Streams

### Work Stream 1: Execution Optimization
**Files:** `execution/` (18 modules)
**Focus:** Order execution, fill prediction, slippage monitoring
**Dependencies:** config, models/risk_manager
**Test Suite:** `test_cancel_optimizer.py`, `test_fill_predictor.py`, `test_slippage_monitor.py`
**Recommended Team Size:** 3-4 agents

### Work Stream 2: LLM Agents & Sentiment
**Files:** `llm/` (40 modules)
**Focus:** Multi-agent consensus, debate mechanism, sentiment analysis
**Dependencies:** config, minimal external
**Test Suite:** `test_multi_agent.py`, `test_debate_mechanism.py`, `test_llm_sentiment.py`
**Recommended Team Size:** 4-5 agents

### Work Stream 3: Market Scanning
**Files:** `scanners/` + `indicators/` (7 modules)
**Focus:** Options scanning, movement detection, technical indicators
**Dependencies:** config, llm (movement_scanner only)
**Test Suite:** `test_options_scanner.py`, `test_unusual_activity_scanner.py`
**Recommended Team Size:** 2-3 agents

### Work Stream 4: Risk Management
**Files:** `models/risk*.py`, `models/circuit_breaker.py`, `compliance/`
**Focus:** Risk models, position limits, trading halts
**Dependencies:** config, observability
**Test Suite:** `test_circuit_breaker.py`, `test_risk_management.py`
**Recommended Team Size:** 1-2 agents (HIGH RISK - careful coordination)

### Work Stream 5: Backtesting & Evaluation
**Files:** `backtesting/` + `evaluation/` (35 modules)
**Focus:** Strategy backtesting, walk-forward analysis, agent evaluation
**Dependencies:** models (loosely coupled)
**Test Suite:** `test_monte_carlo.py`, `test_walk_forward.py`, `test_agent_contest.py`
**Recommended Team Size:** 2-3 agents

### Work Stream 6: Observability & Monitoring
**Files:** `observability/` (16 modules)
**Focus:** Logging, tracing, metrics, anomaly detection
**Dependencies:** None (orthogonal)
**Test Suite:** `test_structured_logging.py`, `test_otel_tracer.py`
**Recommended Team Size:** 2-3 agents

### Work Stream 7: API & Integration
**Files:** `api/` + `mcp/` (13 modules)
**Focus:** REST API, WebSocket, MCP servers
**Dependencies:** execution, models/risk_manager
**Test Suite:** `test_order_queue_api.py`, `test_rest_api.py`
**Recommended Team Size:** 2-3 agents

---

## 3. Coordination System Design

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PUCS COORDINATOR                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Lock      │  │   Intent    │  │   Cross     │             │
│  │   Manager   │  │   Signal    │  │   Logger    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┴────────────────┘                     │
│                          │                                      │
│                   ┌──────▼──────┐                               │
│                   │   State     │                               │
│                   │   Store     │                               │
│                   └──────┬──────┘                               │
└──────────────────────────┼──────────────────────────────────────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     │                     │                     │
┌────▼────┐          ┌─────▼────┐          ┌─────▼────┐
│ Stream  │          │ Stream   │          │ Stream   │
│ Agent 1 │          │ Agent 2  │          │ Agent N  │
└─────────┘          └──────────┘          └──────────┘
```

### 3.2 State Store Schema

```json
{
  "session_id": "PUCS-20251206-001",
  "started_at": "2025-12-06T10:00:00Z",
  "streams": {
    "execution": {
      "status": "active",
      "agents": ["exec-agent-1", "exec-agent-2"],
      "locked_files": [
        {
          "path": "execution/fill_predictor.py",
          "agent": "exec-agent-1",
          "intent": "Adding ML model feature",
          "locked_at": "2025-12-06T10:15:00Z",
          "expires_at": "2025-12-06T10:45:00Z"
        }
      ],
      "completed_tasks": 5,
      "pending_tasks": 12
    }
  },
  "critical_files": {
    "config/__init__.py": {
      "status": "locked",
      "owner": "risk-agent-1",
      "queue": ["llm-agent-3", "exec-agent-1"]
    }
  },
  "conflicts": [],
  "cross_log": []
}
```

### 3.3 Lock Manager

The Lock Manager prevents simultaneous modifications to the same files.

**Lock File Format:**

```json
// .claude/state/file_locks.json
{
  "locks": {
    "llm/sentiment.py": {
      "agent": "agent-llm-1",
      "stream": "llm",
      "acquired": "2025-12-06T10:00:00Z",
      "expires": "2025-12-06T11:00:00Z",
      "task": "Consolidate sentiment to package"
    }
  },
  "version": 1
}
```

**Working Implementation (using fcntl for atomic operations):**

```python
# scripts/agent_lock.py
import json
import fcntl
from datetime import datetime, timedelta
from pathlib import Path

LOCK_FILE = Path(".claude/state/file_locks.json")
LOCK_DURATION_MINUTES = 60

def acquire_lock(file_path: str, agent_id: str, stream: str, task: str) -> bool:
    """Acquire exclusive lock on a file for an agent."""
    with open(LOCK_FILE, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            locks = json.load(f)

            # Check if file is already locked
            if file_path in locks["locks"]:
                lock = locks["locks"][file_path]
                expires = datetime.fromisoformat(lock["expires"])
                if datetime.now() < expires:
                    return False  # Still locked

            # Acquire lock
            locks["locks"][file_path] = {
                "agent": agent_id,
                "stream": stream,
                "acquired": datetime.now().isoformat(),
                "expires": (datetime.now() + timedelta(minutes=LOCK_DURATION_MINUTES)).isoformat(),
                "task": task
            }

            f.seek(0)
            json.dump(locks, f, indent=2)
            f.truncate()
            return True
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def release_lock(file_path: str, agent_id: str) -> bool:
    """Release lock on a file."""
    with open(LOCK_FILE, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            locks = json.load(f)
            if file_path in locks["locks"]:
                if locks["locks"][file_path]["agent"] == agent_id:
                    del locks["locks"][file_path]
                    f.seek(0)
                    json.dump(locks, f, indent=2)
                    f.truncate()
                    return True
            return False
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def check_lock(file_path: str) -> dict | None:
    """Check if a file is locked."""
    with open(LOCK_FILE, "r") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            locks = json.load(f)
            if file_path in locks["locks"]:
                lock = locks["locks"][file_path]
                expires = datetime.fromisoformat(lock["expires"])
                if datetime.now() < expires:
                    return lock
            return None
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

### 3.4 Intent Signal System

Before starting work, agents signal their intent to avoid conflicts:

```python
class IntentSignal:
    """Signals agent's intention to work on specific files/modules."""

    def signal_intent(self, agent_id: str, work_stream: str,
                      files: list[str], description: str,
                      estimated_duration_min: int) -> IntentResult:
        """
        Signal intent to work on files.

        The system checks for conflicts with:
        1. Active locks on any of the files
        2. Other agents' intents overlapping
        3. Dependency violations

        Returns conflicts to resolve before proceeding.
        """
        pass

    def check_conflicts(self, proposed_files: list[str]) -> list[Conflict]:
        """Check for potential conflicts with proposed work."""
        pass

    def negotiate_partition(self, agent_a: str, agent_b: str,
                           shared_module: str) -> PartitionResult:
        """
        Negotiate work partition when two agents need same module.

        Strategy:
        1. Identify specific functions/classes each needs
        2. Propose non-overlapping scopes
        3. If impossible, queue second agent
        """
        pass
```

### 3.5 Cross-Logger (JSONL Append-Only)

All agent actions are logged to shared JSONL files (append-only for safe concurrent writes):

**Log Files:**

```text
.claude/state/
├── agent_activity.jsonl     # All agent actions (append-only)
├── conflict_warnings.jsonl  # Conflict alerts (append-only)
└── handoffs.json           # Handoff context between agents
```

**Activity Log Format:**

```jsonl
{"timestamp": "2025-12-06T10:00:00Z", "agent": "exec-agent-1", "stream": "execution", "action": "started", "task": "Fill predictor enhancement"}
{"timestamp": "2025-12-06T10:01:00Z", "agent": "exec-agent-1", "stream": "execution", "action": "locked", "file": "execution/fill_predictor.py"}
{"timestamp": "2025-12-06T10:30:00Z", "agent": "exec-agent-1", "stream": "execution", "action": "completed", "files_modified": 3, "tests_passed": true}
{"timestamp": "2025-12-06T10:31:00Z", "agent": "exec-agent-1", "stream": "execution", "action": "released", "file": "execution/fill_predictor.py"}
```

**Conflict Warning Log:**

```jsonl
{"timestamp": "2025-12-06T10:15:00Z", "agent": "llm-agent-2", "warning": "news_analyzer.py imports from sentiment.py - llm-agent-1 has lock", "severity": "medium"}
{"timestamp": "2025-12-06T10:20:00Z", "agent": "risk-agent-1", "warning": "volatility_surface.py depends on risk_manager.py - change in progress", "severity": "high"}
```

**Implementation:**

```python
# scripts/cross_logger.py
import json
from datetime import datetime
from pathlib import Path
import fcntl

ACTIVITY_LOG = Path(".claude/state/agent_activity.jsonl")
CONFLICT_LOG = Path(".claude/state/conflict_warnings.jsonl")

def log_activity(agent_id: str, stream: str, action: str, **kwargs) -> None:
    """Append activity to JSONL log (thread-safe)."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_id,
        "stream": stream,
        "action": action,
        **kwargs
    }
    _append_jsonl(ACTIVITY_LOG, entry)

def log_conflict(agent_id: str, warning: str, severity: str = "medium") -> None:
    """Log a potential conflict warning."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_id,
        "warning": warning,
        "severity": severity
    }
    _append_jsonl(CONFLICT_LOG, entry)

def _append_jsonl(file_path: Path, entry: dict) -> None:
    """Append entry to JSONL file with file locking."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(entry) + "\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def get_stream_log(stream: str, since: datetime = None) -> list[dict]:
    """Get all log entries for a work stream."""
    entries = []
    if ACTIVITY_LOG.exists():
        with open(ACTIVITY_LOG, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("stream") == stream:
                    if since is None or datetime.fromisoformat(entry["timestamp"]) >= since:
                        entries.append(entry)
    return entries
```

### 3.6 Handoff System

For sequential dependencies between streams, handoffs provide context:

**Handoff Format:**

```json
// .claude/state/handoffs.json
{
  "handoffs": [
    {
      "from_agent": "risk-agent-1",
      "to_agents": ["exec-agent-1", "api-agent-1"],
      "timestamp": "2025-12-06T11:00:00Z",
      "completed_tasks": ["Risk manager refactoring", "Circuit breaker updates"],
      "context": {
        "new_interfaces": ["RiskEnforcementChain"],
        "breaking_changes": ["RiskManager.validate() signature changed to validate(order, context)"],
        "test_status": "all 47 tests passing",
        "notes": "Dependent modules should update imports and call signatures"
      }
    }
  ]
}
```

**Usage:**

```python
def record_handoff(from_agent: str, to_agents: list[str],
                   completed_tasks: list[str], context: dict) -> None:
    """Record handoff context for dependent agents."""
    handoff_file = Path(".claude/state/handoffs.json")

    with open(handoff_file, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            data = json.load(f)
            data["handoffs"].append({
                "from_agent": from_agent,
                "to_agents": to_agents,
                "timestamp": datetime.now().isoformat(),
                "completed_tasks": completed_tasks,
                "context": context
            })
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

---

## 4. Conflict Detection & Resolution

### 4.1 Conflict Types

| Type | Description | Auto-Resolution |
|------|-------------|-----------------|
| FILE_LOCK | Two agents need same file | Queue second agent |
| DEPENDENCY | Change affects dependent module | Notify dependent agent |
| SEMANTIC | Same function modified differently | Manual merge required |
| RESOURCE | Shared resource (DB, API) contention | Rate limiting |

### 4.2 Resolution Strategies

```python
class ConflictResolver:
    """Resolves conflicts between parallel agents."""

    STRATEGIES = {
        "QUEUE": "Second agent waits for first to complete",
        "PARTITION": "Split work into non-overlapping scopes",
        "MERGE": "Attempt automatic merge of changes",
        "ARBITRATE": "Coordinator decides which change wins",
        "ROLLBACK": "Revert one agent's changes"
    }

    def resolve(self, conflict: Conflict) -> Resolution:
        """
        Resolve a detected conflict.

        Resolution flow:
        1. Check if auto-resolution possible
        2. If PARTITION possible, propose split
        3. If MERGE possible, attempt merge
        4. Otherwise, QUEUE or ARBITRATE
        """
        pass

    def auto_merge(self, file_path: str,
                   change_a: Change, change_b: Change) -> MergeResult:
        """
        Attempt automatic merge of two changes.

        Only works if:
        - Changes are to different functions/classes
        - Changes are additive (not modifying same lines)
        - No semantic conflicts detected
        """
        pass
```

### 4.3 Dependency Violation Detection

```python
class DependencyChecker:
    """Detects changes that might break dependent modules."""

    def check_impact(self, changed_file: str,
                     changes: list[Change]) -> ImpactReport:
        """
        Analyze impact of changes on dependent modules.

        Checks:
        1. Function signature changes
        2. Removed/renamed exports
        3. Changed return types
        4. Modified class hierarchies
        """
        pass

    def notify_dependents(self, impact: ImpactReport) -> None:
        """Notify agents working on dependent modules."""
        pass
```

---

## 5. Git Integration

### 5.1 Git Worktree Strategy (Preferred)

**Why Worktrees?** Each parallel stream gets its own working directory, providing true isolation with no merge conflicts until explicit merge:

```bash
# Create worktrees for parallel streams (from main repo)
git worktree add ../trading-bot-stream-exec feature/stream-execution
git worktree add ../trading-bot-stream-llm feature/stream-llm
git worktree add ../trading-bot-stream-scan feature/stream-scanners
git worktree add ../trading-bot-stream-risk feature/stream-risk
git worktree add ../trading-bot-stream-back feature/stream-backtesting
git worktree add ../trading-bot-stream-obs feature/stream-observability
git worktree add ../trading-bot-stream-api feature/stream-api
```

**Directory Structure:**

```
/home/user/projects/
├── Claude_code_Quantconnect_trading_bot/  (main repo)
├── trading-bot-stream-exec/               (execution worktree)
├── trading-bot-stream-llm/                (llm worktree)
├── trading-bot-stream-scan/               (scanners worktree)
└── ...
```

### 5.2 Branch Naming Convention

```
feature/stream-{NAME}-{description}
├── feature/stream-execution-fill-predictor
├── feature/stream-llm-sentiment-analysis
├── feature/stream-risk-circuit-breaker
└── ...
```

### 5.3 Commit Protocol

```bash
# Format: [PUCS-<stream>] <type>: <description>
# Example commits:

[PUCS-execution] feat: Add ML model to fill predictor
[PUCS-llm] fix: Correct sentiment analysis threshold
[PUCS-risk] refactor: Simplify circuit breaker logic

# Merge strategy: Each stream merges to develop via PR
# Use --no-ff to preserve stream history
```

### 5.4 Merge Order (Dependency-Based)

```
Phase 1 (Parallel - No Dependencies):
  └── Streams: observability, scanners, backtesting

Phase 2 (After Phase 1):
  └── Streams: execution, llm, risk

Phase 3 (After Phase 2):
  └── Streams: api

Phase 4 (After All):
  └── Applications: algorithms, ui, mcp
```

### 5.3 Automated Conflict Prevention

```yaml
# .github/workflows/pucs-check.yml
name: PUCS Conflict Check

on:
  pull_request:
    branches: ['pucs/*']

jobs:
  conflict-check:
    runs-on: ubuntu-latest
    steps:
      - name: Check for file conflicts
        run: |
          # Get list of modified files
          FILES=$(git diff --name-only origin/develop...HEAD)

          # Check against PUCS state store
          python scripts/pucs_conflict_check.py --files "$FILES"

      - name: Validate stream boundaries
        run: |
          # Ensure changes stay within stream boundaries
          python scripts/pucs_boundary_check.py

      - name: Run stream-specific tests
        run: |
          pytest tests/ -m "$STREAM_NAME" -v
```

---

## 6. Agent Coordination Protocol

### 6.1 Agent Lifecycle

```
┌──────────────────────────────────────────────────────────┐
│                    AGENT LIFECYCLE                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. REGISTER                                             │
│     └── Register with coordinator, receive agent_id     │
│                                                          │
│  2. CLAIM_STREAM                                         │
│     └── Request assignment to work stream               │
│                                                          │
│  3. SIGNAL_INTENT                                        │
│     └── Declare files/modules to work on                │
│                                                          │
│  4. ACQUIRE_LOCKS                                        │
│     └── Lock files before modification                  │
│                                                          │
│  5. WORK                                                 │
│     └── Perform development tasks                       │
│     └── Log all actions to cross-logger                 │
│                                                          │
│  6. COMMIT                                               │
│     └── Commit changes to stream branch                 │
│                                                          │
│  7. RELEASE_LOCKS                                        │
│     └── Release all held locks                          │
│                                                          │
│  8. REPORT                                               │
│     └── Report completion status to coordinator         │
│                                                          │
│  9. HANDOFF (optional)                                   │
│     └── Transfer work to another agent if needed        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 6.2 Communication Protocol

```python
# Agent-to-Coordinator messages

class AgentMessage:
    REGISTER = "register"           # Agent starting up
    CLAIM_STREAM = "claim_stream"   # Request stream assignment
    SIGNAL_INTENT = "signal_intent" # Declare work scope
    ACQUIRE_LOCK = "acquire_lock"   # Request file lock
    RELEASE_LOCK = "release_lock"   # Release file lock
    LOG_ACTION = "log_action"       # Log to cross-logger
    REPORT_CONFLICT = "report_conflict"  # Report detected conflict
    REQUEST_HELP = "request_help"   # Need coordinator intervention
    COMPLETE_TASK = "complete_task" # Task finished
    HANDOFF = "handoff"             # Transfer to another agent

# Coordinator-to-Agent messages

class CoordinatorMessage:
    REGISTERED = "registered"       # Registration confirmed
    STREAM_ASSIGNED = "stream_assigned"  # Stream assignment
    LOCK_GRANTED = "lock_granted"   # Lock request approved
    LOCK_DENIED = "lock_denied"     # Lock request denied (with reason)
    CONFLICT_ALERT = "conflict_alert"  # Potential conflict detected
    TASK_ASSIGNED = "task_assigned" # New task to work on
    PAUSE = "pause"                 # Stop current work (conflict resolution)
    RESUME = "resume"               # Continue after pause
    TERMINATE = "terminate"         # Agent should stop
```

---

## 7. Monitoring Dashboard

### 7.1 Real-Time Status Display

```
╔══════════════════════════════════════════════════════════════════╗
║              PUCS MONITORING DASHBOARD                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  SESSION: PUCS-20251206-001  │  STARTED: 10:00 AM  │  3h 45m    ║
║                                                                  ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │  WORK STREAMS STATUS                                        │  ║
║  ├────────────────────────────────────────────────────────────┤  ║
║  │  execution     ██████████████░░░░░░  70%  [3 agents]       │  ║
║  │  llm           █████████░░░░░░░░░░░  45%  [4 agents]       │  ║
║  │  scanners      ████████████████████  100% [2 agents] ✓     │  ║
║  │  risk          ███████████████░░░░░  75%  [2 agents]       │  ║
║  │  backtesting   ██████░░░░░░░░░░░░░░  30%  [2 agents]       │  ║
║  │  observability ████████████████░░░░  80%  [2 agents]       │  ║
║  │  api           █████████████░░░░░░░  65%  [3 agents]       │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │  ACTIVE LOCKS                                               │  ║
║  ├────────────────────────────────────────────────────────────┤  ║
║  │  config/__init__.py     │  risk-agent-1   │  25 min left   │  ║
║  │  execution/fill_pred.py │  exec-agent-2   │  15 min left   │  ║
║  │  llm/agents/base.py     │  llm-agent-1    │  40 min left   │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │  RECENT ACTIVITY (last 10 entries)                         │  ║
║  ├────────────────────────────────────────────────────────────┤  ║
║  │  13:42  exec-agent-1   CHANGE  Modified fill_predictor.py  │  ║
║  │  13:40  llm-agent-3    INFO    Starting sentiment analysis │  ║
║  │  13:38  risk-agent-1   CHANGE  Updated circuit_breaker.py  │  ║
║  │  13:35  obs-agent-2    INFO    Added new metrics endpoint  │  ║
║  │  ...                                                       │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  CONFLICTS: 0  │  QUEUED: 2  │  COMMITS: 47  │  TESTS: ✓ ALL   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 8. Implementation Plan

### Phase 1: Foundation (Week 1)
- [ ] Implement State Store with file-based persistence
- [ ] Create Lock Manager with basic EXCLUSIVE locks
- [ ] Set up Cross-Logger infrastructure
- [ ] Define stream boundaries in configuration

### Phase 2: Coordination (Week 2)
- [ ] Implement Intent Signal system
- [ ] Add conflict detection algorithms
- [ ] Create basic conflict resolution (QUEUE strategy)
- [ ] Build agent communication protocol

### Phase 3: Git Integration (Week 3)
- [ ] Set up branch structure automation
- [ ] Create PR templates for streams
- [ ] Implement automated conflict checks
- [ ] Add stream-specific test runners

### Phase 4: Monitoring (Week 4)
- [ ] Build monitoring dashboard
- [ ] Add real-time status updates
- [ ] Create alerting for conflicts
- [ ] Implement session reports

### Phase 5: Advanced Features (Week 5+)
- [ ] Add PARTITION conflict resolution
- [ ] Implement auto-merge capabilities
- [ ] Create dependency impact analysis
- [ ] Build coordinator arbitration logic

---

## 9. State Initialization

Before starting a PUCS session, initialize the required state files:

```bash
#!/bin/bash
# scripts/init_pucs_state.sh

mkdir -p .claude/state/logs

# Initialize file locks
echo '{"locks": {}, "version": 1}' > .claude/state/file_locks.json

# Initialize orchestrator state
echo '{"streams": {}, "started_at": null}' > .claude/state/orchestrator_state.json

# Initialize handoffs
echo '{"handoffs": []}' > .claude/state/handoffs.json

# Create empty activity logs (JSONL - append-only)
touch .claude/state/agent_activity.jsonl
touch .claude/state/conflict_warnings.jsonl

echo "PUCS state initialized!"
```

**Required State Files:**

```text
.claude/state/
├── file_locks.json          # Current file locks
├── agent_activity.jsonl     # Activity log (append-only)
├── conflict_warnings.jsonl  # Conflict warnings (append-only)
├── handoffs.json           # Handoff context
├── orchestrator_state.json # Orchestrator state
└── logs/
    ├── exec-agent-1.log
    ├── llm-agent-1.log
    └── ...
```

---

## 10. Usage Example

### Starting a PUCS Session

```bash
# 1. Initialize state files
./scripts/init_pucs_state.sh

# 2. Initialize a new PUCS session
python scripts/pucs.py init --session "PUCS-20251206-001"

# 2. Launch parallel agents for each stream
python scripts/pucs.py launch-stream execution --agents 3
python scripts/pucs.py launch-stream llm --agents 4
python scripts/pucs.py launch-stream scanners --agents 2
# ... etc

# 3. Monitor progress
python scripts/pucs.py dashboard

# 4. View cross-log
python scripts/pucs.py log --stream all --since "1h"

# 5. Check for conflicts
python scripts/pucs.py conflicts --unresolved

# 6. Finalize session (merge all streams)
python scripts/pucs.py finalize
```

### Agent Commands (from within an agent)

```python
from pucs import PUCSClient

# Initialize client
pucs = PUCSClient(agent_id="exec-agent-1", stream="execution")

# Signal intent before starting work
intent_result = pucs.signal_intent(
    files=["execution/fill_predictor.py", "execution/fill_ml_model.py"],
    description="Adding gradient boosting model for fill prediction",
    estimated_duration_min=45
)

if intent_result.conflicts:
    # Handle conflicts
    for conflict in intent_result.conflicts:
        print(f"Conflict: {conflict.file} - {conflict.holder}")
else:
    # Acquire locks and proceed
    locks = pucs.acquire_locks(intent_result.files)

    # ... do work ...

    # Log changes
    pucs.log_change(
        action="Modified fill_predictor.py",
        files=["execution/fill_predictor.py"],
        details={"lines_added": 120, "functions_added": 3}
    )

    # Release locks when done
    pucs.release_locks()
```

---

## 10. Configuration

### pucs_config.yaml

```yaml
# PUCS Configuration

session:
  max_duration_hours: 8
  auto_finalize: false

streams:
  execution:
    files_pattern: "execution/**/*.py"
    max_agents: 4
    dependencies: ["config", "models"]
    test_pattern: "tests/test_*execution*.py"

  llm:
    files_pattern: "llm/**/*.py"
    max_agents: 5
    dependencies: ["config"]
    test_pattern: "tests/test_*llm*.py"

  scanners:
    files_pattern: "scanners/**/*.py,indicators/**/*.py"
    max_agents: 3
    dependencies: ["config", "llm"]
    test_pattern: "tests/test_*scanner*.py"

  risk:
    files_pattern: "models/risk*.py,models/circuit*.py,compliance/**/*.py"
    max_agents: 2
    critical: true  # Requires extra review
    dependencies: ["config"]
    test_pattern: "tests/test_*risk*.py,tests/test_circuit*.py"

  backtesting:
    files_pattern: "backtesting/**/*.py,evaluation/**/*.py"
    max_agents: 3
    dependencies: ["models"]
    test_pattern: "tests/test_*backtest*.py"

  observability:
    files_pattern: "observability/**/*.py"
    max_agents: 3
    dependencies: []
    test_pattern: "tests/test_*observability*.py"

  api:
    files_pattern: "api/**/*.py,mcp/**/*.py"
    max_agents: 3
    dependencies: ["execution", "models"]
    test_pattern: "tests/test_*api*.py"

critical_files:
  - path: "config/__init__.py"
    max_lock_duration_min: 30
    requires_review: true

  - path: "models/circuit_breaker.py"
    max_lock_duration_min: 45
    requires_review: true
    test_required: true

  - path: "models/risk_manager.py"
    max_lock_duration_min: 45
    requires_review: true
    test_required: true

locks:
  default_duration_min: 30
  max_duration_min: 120
  queue_timeout_min: 60

logging:
  level: INFO
  persist_path: ".claude/state/pucs_log.json"
  max_entries: 10000

conflict_resolution:
  auto_merge: true
  partition_enabled: true
  arbitration_timeout_min: 15
```

---

## 11. Best Practices

### For Parallel Agents

1. **Always signal intent before starting work**
   - Prevents conflicts and enables coordination

2. **Acquire minimal locks**
   - Lock only files you're actively modifying
   - Release locks as soon as possible

3. **Stay within stream boundaries**
   - Don't modify files outside your assigned stream
   - Request cross-stream work through coordinator

4. **Log all significant actions**
   - Enables debugging and coordination
   - Helps other agents understand changes

5. **Run tests before releasing locks**
   - Ensures changes don't break the stream
   - Catches issues early

### For Session Coordinators

1. **Monitor conflict rates**
   - High conflict rate indicates poor stream partitioning
   - Consider adjusting boundaries

2. **Balance agent counts**
   - More agents ≠ faster completion
   - Optimal is typically 2-4 per stream

3. **Review critical file changes**
   - Changes to critical files affect multiple streams
   - Require human review before merge

4. **Finalize incrementally**
   - Merge completed streams early
   - Don't wait for all streams to finish

---

## 12. References

### Industry Best Practices

- [Microsoft AI Agent Design Patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
- [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Google Agent Development Kit](https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/)
- [Agent-MCP Framework](https://github.com/rinadelph/Agent-MCP)
- [Git Branching Strategies](https://www.abtasty.com/blog/git-branching-strategies/)
- [Agentic Swarm Coding](https://www.augmentcode.com/guides/what-is-agentic-swarm-coding-definition-architecture-and-use-cases)
- [Simon Willison: Parallel Coding Agent Lifestyle](https://simonwillison.net/2025/Oct/5/parallel-coding-agents/)
- [AI Native Dev: Parallelizing AI Coding Agents](https://ainativedev.io/news/how-to-parallelize-ai-coding-agents)
- [Metacircuits: Managing Parallel Coding Agents](https://metacircuits.substack.com/p/managing-parallel-coding-agents-without)
- [Parallel AI Development with Git Worktrees](https://medium.com/@ooi_yee_fei/parallel-ai-development-with-git-worktrees-f2524afc3e33)
- [Solving Parallel Workflow Conflicts](https://raminmammadzada.medium.com/solving-parallel-workflow-conflicts-between-ai-agents-and-developers-in-shared-codebases-286504422125)
- [Claude Flow Framework](https://github.com/ruvnet/claude-flow)
- [Running Claude Agents in Parallel](https://www.curiouslychase.com/ai-development/running-claude-agents-in-parallel-with-git-worktrees)

### Project Documentation

- [docs/ARCHITECTURE.md](ARCHITECTURE.md) - Project architecture
- [docs/PARALLEL_AGENT_COORDINATION.md](PARALLEL_AGENT_COORDINATION.md) - Original coordination doc (superseded)
- [.claude/hooks/agents/agent_orchestrator.py](../.claude/hooks/agents/agent_orchestrator.py) - Existing agent orchestration
- [.claude/RIC_CONTEXT.md](../.claude/RIC_CONTEXT.md) - RIC Loop integration

---

## Appendix A: Work Stream Task Breakdown

### Stream 1: Execution (15 parallelizable tasks)
1. Fill predictor ML model improvements
2. Slippage monitoring enhancements
3. Liquidity scoring algorithm
4. Smart execution cancel/replace logic
5. Profit-taking threshold optimization
6. Spread analysis improvements
7. Arbitrage executor refinements
8. Execution quality metrics
9. Pre-trade validator enhancements
10. Recurring order manager updates
11. Bot-managed positions improvements
12. Option strategies executor
13. Manual legs executor
14. Cancel optimizer
15. Fill ML model training

### Stream 2: LLM Agents (20 parallelizable tasks)
1. Fine-tune trader persona
2. Fine-tune analyst persona
3. Fine-tune risk manager persona
4. Improve debate mechanism convergence
5. Add new sentiment signals
6. Optimize ensemble weights
7. Add derivatives specialist agent
8. Improve news analyzer
9. Enhance emotion detector
10. Reddit sentiment improvements
11. Entity extractor updates
12. Signal aggregator refinements
13. Cost optimization improvements
14. Prompt optimizer updates
15. Model router enhancements
16. Guardrails improvements
17. Safe agent wrapper updates
18. Reasoning logger enhancements
19. Multi-agent consensus improvements
20. Supervisor orchestration updates

### (Additional streams have similar breakdowns)

---

**Document Status:** Proposal - Awaiting Review
**Next Steps:** Implement Phase 1 (Foundation)
