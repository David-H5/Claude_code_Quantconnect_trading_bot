# Parallel Upgrade Coordination System (PUCS)

**Version:** 2.0 (Unified Framework)
**Date:** December 7, 2025
**Status:** Active
**Supersedes:** v1.1, PARALLEL_AGENT_COORDINATION.md

> **Key Change in v2.0:** This version integrates with existing codebase infrastructure
> instead of proposing duplicate systems. All coordination now leverages the existing
> agent orchestrator, logging, and state management systems.

---

## Executive Summary

The **Parallel Upgrade Coordination System (PUCS)** enables multiple AI agents to work on the codebase simultaneously without conflicts. This unified framework integrates with existing infrastructure:

| Component | Existing System | PUCS Extension |
|-----------|-----------------|----------------|
| Orchestration | `agent_orchestrator.py` | Work stream routing |
| Logging | `AgentLogger` | Stream-aware logging |
| State | `OvernightStateManager` | File lock tracking |
| Handoffs | `.claude/state/handoff.json` | Multi-agent handoffs |
| Circuit Breaker | `AgentCircuitBreaker` | Stream-level breakers |
| Tracing | `Tracer`, `LLMTracer` | Stream correlation |

---

## 0. Prerequisites: Codebase Consolidation

> **IMPORTANT**: Before implementing PUCS, the following consolidation tasks MUST be completed.
> These resolve existing conflicts that would otherwise be amplified by PUCS.

### 0.1 Consolidation Checklist

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│              PUCS PREREQUISITE CONSOLIDATION CHECKLIST                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: BLOCKING ISSUES (Must complete before PUCS)          Est: 2 days  │
│  ──────────────────────────────────────────────────────────────────────────  │
│  [ ] P0-1. Merge DecisionLogger + DecisionTracer                             │
│  [ ] P0-2. Consolidate State Managers (3 → 1)                                │
│  [ ] P0-3. Delete deprecated models/retry_handler.py                         │
│  [ ] P0-4. Create BaseCircuitBreaker shared class                            │
│                                                                              │
│  PHASE 2: RECOMMENDED FIXES (Should complete)                   Est: 2 days  │
│  ──────────────────────────────────────────────────────────────────────────  │
│  [ ] P1-1. Create shared ValidationResult class                              │
│  [ ] P1-2. Use canonical parse_progress_file() everywhere                    │
│  [ ] P1-3. Consolidate webhook configurations                                │
│  [ ] P1-4. Delete config/watchdog.json (duplicate)                           │
│  [ ] P1-5. Create shared env_expansion.py utility                            │
│  [ ] P1-6. Delete llm/tools/finbert.py (duplicate)                           │
│                                                                              │
│  PHASE 3: CLEANUP (Can run parallel with PUCS)                  Est: 1 day   │
│  ──────────────────────────────────────────────────────────────────────────  │
│  [ ] P2-1. Consolidate path configurations                                   │
│  [ ] P2-2. Merge logging configurations                                      │
│  [ ] P2-3. Organize state files with registry                                │
│  [ ] P2-4. Update deprecated module imports                                  │
│  [ ] P2-5. Delete deprecated re-export modules                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 0.2 Phase 1: Blocking Issues (P0)

These MUST be resolved before PUCS implementation to prevent creating additional duplicate systems.

#### P0-1. Merge DecisionLogger + DecisionTracer

**Problem**: Two systems track agent decisions with different data structures.

| System | File | Output |
|--------|------|--------|
| DecisionLogger | `llm/decision_logger.py` (727 lines) | `decision_logs/*.json` |
| DecisionTracer | `observability/decision_tracer.py` (352 lines) | `logs/decisions/*.json` |

**Conflict Details**:

- THREE different `ReasoningStep` definitions exist in codebase
- Different output directories (`decision_logs/` vs `logs/decisions/`)
- Different outcome enums (PENDING/EXECUTED/REJECTED vs TIMEOUT/ERROR/OVERRIDDEN)
- `ReasoningLogger` in `llm/reasoning_logger.py` has optional (not mandatory) link

**Resolution Steps**:

```bash
# Step 1: Enhance DecisionLogger with DecisionTracer features
# Add to llm/decision_logger.py:
# - inputs/outputs fields to ReasoningStep
# - timestamp per reasoning step
# - export_to_file() method matching DecisionTracer

# Step 2: Update imports
grep -r "from observability.decision_tracer import" --include="*.py" | \
  xargs sed -i 's/from observability.decision_tracer import/from llm.decision_logger import/g'

# Step 3: Delete DecisionTracer
rm observability/decision_tracer.py

# Step 4: Update observability/logging/adapters/decision.py to use DecisionLogger only
```

**Verification**:

```bash
# Ensure no remaining imports
grep -r "decision_tracer" --include="*.py"
# Run tests
pytest tests/test_decision_logger.py -v
```

---

#### P0-2. Consolidate State Managers

**Problem**: Three separate state managers with overlapping data.

| Manager | File | State File | Has Locking |
|---------|------|------------|-------------|
| SessionStateManager | `scripts/session_state_manager.py` | `logs/session_state.json` | No |
| OvernightStateManager | `utils/overnight_state.py` | `logs/overnight_state.json` | Yes (fcntl) |
| RICStateManager | `.claude/hooks/core/ric.py:6198` | `.claude/state/ric.json` | No |

**Resolution Steps**:

```bash
# Step 1: Keep OvernightStateManager as canonical (has file locking)
# File: utils/overnight_state.py

# Step 2: Update SessionStateManager to delegate to OvernightStateManager
# In scripts/session_state_manager.py, add deprecation:
```

```python
# scripts/session_state_manager.py - UPDATE TO:
"""DEPRECATED: Use utils.overnight_state.OvernightStateManager instead."""
import warnings
from utils.overnight_state import OvernightStateManager

warnings.warn(
    "SessionStateManager is deprecated. Use OvernightStateManager from utils.overnight_state",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backwards compatibility
SessionStateManager = OvernightStateManager
```

```bash
# Step 3: Update RICStateManager to NOT duplicate session tracking
# RICStateManager should ONLY manage RIC-specific state (phase, insights, throttles)
# Session-level state should read from OvernightStateManager

# Step 4: Update imports
grep -r "from scripts.session_state_manager import" --include="*.py"
# Update each to use: from utils.overnight_state import OvernightStateManager
```

**Verification**:

```bash
pytest tests/test_overnight_state.py -v
```

---

#### P0-3. Delete Deprecated Retry Handler

**Problem**: `models/retry_handler.py` is deprecated but has same-named classes.

| Module | Status | Replacement |
|--------|--------|-------------|
| `models/retry_handler.py` | DEPRECATED | `utils.error_handling` |
| `agent_orchestrator.py:90-159` | ACTIVE | Different purpose (agents) |

**Resolution Steps**:

```bash
# Step 1: Find all imports
grep -r "from models.retry_handler import" --include="*.py"
grep -r "from models import retry_handler" --include="*.py"

# Step 2: Update each import to use utils.error_handling
# Change: from models.retry_handler import RetryConfig, retry_with_backoff
# To:     from utils.error_handling import RetryConfig, retry_with_backoff

# Step 3: Delete the deprecated module
rm models/retry_handler.py

# Step 4: Update models/__init__.py if it exports retry_handler
```

**Verification**:

```bash
pytest tests/test_error_handling.py -v
python -c "from utils.error_handling import RetryConfig, retry_with_backoff; print('OK')"
```

---

#### P0-4. Create BaseCircuitBreaker Shared Class

**Problem**: Two circuit breakers with no shared code.

| Breaker | File | Purpose |
|---------|------|---------|
| TradingCircuitBreaker | `models/circuit_breaker.py:102` | Trading risk limits |
| AgentCircuitBreaker | `agent_orchestrator.py:224` | Agent failure tracking |

**Resolution Steps**:

```python
# Create: models/base_circuit_breaker.py

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Tripped, blocking
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class BaseCircuitBreakerConfig:
    """Shared configuration for all circuit breakers."""
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 300
    half_open_max_calls: int = 3

class BaseCircuitBreaker(ABC):
    """Abstract base for all circuit breaker implementations."""

    def __init__(self, config: BaseCircuitBreakerConfig):
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        return self._state

    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            return self._should_attempt_reset()
        if self._state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.config.half_open_max_calls
        return False

    def record_success(self) -> None:
        """Record successful execution."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.config.half_open_max_calls:
                self._reset()
        self._failure_count = 0

    def record_failure(self) -> None:
        """Record failed execution."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        if self._failure_count >= self.config.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        """Open the circuit breaker."""
        self._state = CircuitState.OPEN
        self._on_trip()

    def _reset(self) -> None:
        """Reset to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0
        self._on_reset()

    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to attempt recovery."""
        if self._last_failure_time is None:
            return True
        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        if elapsed >= self.config.recovery_timeout_seconds:
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            return True
        return False

    @abstractmethod
    def _on_trip(self) -> None:
        """Hook called when circuit trips. Override for specific behavior."""
        pass

    @abstractmethod
    def _on_reset(self) -> None:
        """Hook called when circuit resets. Override for specific behavior."""
        pass
```

```bash
# Step 2: Update TradingCircuitBreaker to extend BaseCircuitBreaker
# Step 3: Update AgentCircuitBreaker to extend BaseCircuitBreaker
# Step 4: PUCS StreamCircuitBreaker will extend AgentCircuitBreaker
```

---

### 0.3 Phase 2: Recommended Fixes (P1)

These should be completed to avoid confusion during PUCS implementation.

#### P1-1. Create Shared ValidationResult Class

**Problem**: 4 separate `ValidationResult` classes.

**Solution**:

```python
# Create: models/validation.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationIssue:
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    location: str | None = None
    suggestion: str | None = None

@dataclass
class ValidationResult:
    """Unified validation result for all validators."""
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)
```

**Migration**: Update all 4 files to import from `models.validation`.

---

#### P1-2. Use Canonical parse_progress_file()

**Problem**: 4 implementations of `parse_progress_file()`.

**Solution**:

```bash
# Canonical location: utils/progress_parser.py

# Update these files to import from canonical location:
# 1. .claude/hooks/core/session_stop.py
# 2. .claude/hooks/validation/validate_category_docs.py
# 3. scripts/preload_upgrade_guide.py

# Add to each:
from utils.progress_parser import parse_progress_file
# Remove local implementation
```

---

#### P1-3 to P1-6. Quick Fixes

| Task | Action | Command |
|------|--------|---------|
| P1-3 | Consolidate webhooks | Move all to `config/__init__.py:NotificationConfig` |
| P1-4 | Delete watchdog.json | `rm config/watchdog.json` |
| P1-5 | Create env_expansion.py | `touch utils/env_expansion.py` with unified pattern |
| P1-6 | Delete FinBERT duplicate | `rm llm/tools/finbert.py` |

---

### 0.4 Phase 3: Cleanup (P2)

Can run in parallel with PUCS implementation.

#### Deprecated Modules to Update

| Deprecated Module | New Location | Files Using It |
|-------------------|--------------|----------------|
| `evaluation/agent_metrics.py` | `observability.metrics.collectors.agent` | 4 files |
| `utils/alerting_service.py` | `observability.alerting.service` | 3 files |
| `utils/storage_monitor.py` | `observability.monitoring.system.storage` | 1 file |
| `models/correlation_monitor.py` | `observability.monitoring.trading.correlation` | TBD |
| `models/var_monitor.py` | `observability.monitoring.trading.var` | TBD |
| `models/greeks_monitor.py` | `observability.monitoring.trading.greeks` | TBD |

**Migration Script**:

```bash
#!/bin/bash
# scripts/migrate_deprecated_imports.sh

# Update evaluation/agent_metrics imports
find . -name "*.py" -exec sed -i \
  's/from evaluation.agent_metrics import/from observability.metrics.collectors.agent import/g' {} \;

# Update utils/alerting_service imports
find . -name "*.py" -exec sed -i \
  's/from utils.alerting_service import/from observability.alerting.service import/g' {} \;

# Update utils/storage_monitor imports
find . -name "*.py" -exec sed -i \
  's/from utils.storage_monitor import/from observability.monitoring.system.storage import/g' {} \;

echo "Import migration complete. Run tests to verify."
```

---

### 0.5 Verification

After completing consolidation, verify with:

```bash
# 1. No duplicate decision tracking
grep -r "DecisionTracer" --include="*.py" | grep -v "test_" | wc -l  # Should be 0

# 2. No deprecated retry imports
grep -r "from models.retry_handler" --include="*.py" | wc -l  # Should be 0

# 3. State managers consolidated
python -c "from utils.overnight_state import OvernightStateManager; print('OK')"

# 4. All tests pass
pytest tests/ -v --tb=short

# 5. No circular imports
python -c "import config; import utils.overnight_state; import llm.decision_logger; print('No circular imports')"
```

---

### 0.6 Consolidation vs PUCS Timeline

```text
Week 1                          Week 2                          Week 3
├─────────────────────────────┼─────────────────────────────────┼──────────────────────────────┤
│ Phase 1 (P0)                │ Phase 2 (P1)                    │ PUCS Implementation          │
│ ├─ Merge DecisionLogger     │ ├─ ValidationResult class       │ ├─ Stream Lock Manager       │
│ ├─ Consolidate State Mgrs   │ ├─ parse_progress_file()        │ ├─ Intent Signal System      │
│ ├─ Delete retry_handler     │ ├─ Webhook consolidation        │ ├─ PUCS CLI                  │
│ └─ BaseCircuitBreaker       │ └─ Quick fixes                  │ └─ Git worktree setup        │
│                             │                                 │                              │
│ ─────────── GATE ─────────► │ ─────────── GATE ─────────────► │                              │
│ Must pass before Week 2     │ Recommended before Week 3       │ Phase 3 runs parallel        │
└─────────────────────────────┴─────────────────────────────────┴──────────────────────────────┘
```

---

## 1. Architecture Integration

### 1.1 System Hierarchy

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXISTING: Agent Orchestrator v1.5                     │
│               (.claude/hooks/agents/agent_orchestrator.py)               │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │ ModelSelector    │  │ AgentCircuit     │  │ RetryConfig +    │       │
│  │ (Haiku/Sonnet/   │  │ Breaker          │  │ FallbackRouter   │       │
│  │ Opus)            │  │ (failure mgmt)   │  │ (resilience)     │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │ Tracer           │  │ TokenTracker     │  │ ResearchPersister│       │
│  │ (ExecutionTrace) │  │ (cost tracking)  │  │ (docs/research/) │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   NEW: PUCS Extension   │
                    │   (Work Stream Layer)   │
                    └────────────┬────────────┘
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     │                           │                           │
┌────▼────┐     ┌────────────────▼────────────────┐     ┌────▼────┐
│ Stream  │     │     EXISTING: AgentLogger       │     │ Stream  │
│ Lock    │     │  (observability/logging/agent.py)    │ Intent  │
│ Manager │     └─────────────────────────────────┘     │ Signal  │
└─────────┘                                             └─────────┘
```

### 1.2 Codebase Analysis Summary

Based on analysis of **30,587 lines of code** across **150+ modules**:

| Category | Files | Lines | Independence | Existing Orchestration |
|----------|-------|-------|--------------|------------------------|
| Execution | 18 | ~3,000 | 95% | None |
| LLM/Agents | 40 | ~8,000 | 90% | `multi_agent_consensus.py` |
| Scanners | 5 | ~1,200 | 98% | None |
| Models | 22 | ~4,500 | 85% | None |
| Evaluation | 30+ | ~5,000 | 100% | `orchestration_pipeline.py` |
| Observability | 16 | ~2,500 | 100% | None |
| API | 10+ | ~2,000 | 90% | None |

---

## 2. Work Streams

### 2.1 Stream Definitions

Seven independent work streams, each mapped to existing orchestrator capabilities:

| Stream ID | Directory Pattern | Agent Templates | Dependencies |
|-----------|-------------------|-----------------|--------------|
| `EXEC` | `execution/**/*.py` | `implementer`, `refactorer` | config, models |
| `LLM` | `llm/**/*.py` | `architect`, `implementer` | config |
| `SCAN` | `scanners/**/*.py`, `indicators/**/*.py` | `implementer` | config, llm |
| `RISK` | `models/risk*.py`, `compliance/**/*.py` | `risk_reviewer`, `implementer` | config |
| `BACK` | `backtesting/**/*.py`, `evaluation/**/*.py` | `implementer`, `test_analyzer` | models |
| `OBS` | `observability/**/*.py` | `implementer` | None |
| `API` | `api/**/*.py`, `mcp/**/*.py` | `implementer`, `security_scanner` | execution, models |

### 2.2 Stream Configuration

```yaml
# config/pucs_streams.yaml

streams:
  EXEC:
    name: "Execution Optimization"
    files_pattern: "execution/**/*.py"
    max_agents: 4
    agent_templates: ["implementer", "refactorer", "test_analyzer"]
    dependencies: ["config", "models"]
    critical_files:
      - "execution/smart_execution.py"
      - "execution/profit_taking.py"
    test_pattern: "tests/test_*execution*.py"

  LLM:
    name: "LLM Agents & Sentiment"
    files_pattern: "llm/**/*.py"
    max_agents: 5
    agent_templates: ["architect", "implementer", "security_scanner"]
    dependencies: ["config"]
    critical_files:
      - "llm/agents/base.py"
      - "llm/agents/supervisor.py"
    test_pattern: "tests/test_*llm*.py"

  SCAN:
    name: "Market Scanning"
    files_pattern: "scanners/**/*.py,indicators/**/*.py"
    max_agents: 3
    agent_templates: ["implementer"]
    dependencies: ["config", "llm"]
    test_pattern: "tests/test_*scanner*.py"

  RISK:
    name: "Risk Management"
    files_pattern: "models/risk*.py,models/circuit*.py,compliance/**/*.py"
    max_agents: 2
    critical: true
    agent_templates: ["risk_reviewer", "implementer"]
    dependencies: ["config"]
    test_pattern: "tests/test_*risk*.py"

  BACK:
    name: "Backtesting & Evaluation"
    files_pattern: "backtesting/**/*.py,evaluation/**/*.py"
    max_agents: 3
    agent_templates: ["implementer", "test_analyzer"]
    dependencies: ["models"]
    test_pattern: "tests/test_*backtest*.py"

  OBS:
    name: "Observability"
    files_pattern: "observability/**/*.py"
    max_agents: 3
    agent_templates: ["implementer"]
    dependencies: []
    test_pattern: "tests/test_*observability*.py"

  API:
    name: "API & Integration"
    files_pattern: "api/**/*.py,mcp/**/*.py"
    max_agents: 3
    agent_templates: ["implementer", "security_scanner"]
    dependencies: ["execution", "models"]
    test_pattern: "tests/test_*api*.py"
```

---

## 3. Integration with Existing Systems

### 3.1 AgentLogger Integration (NOT a new logger)

PUCS extends the **existing** `AgentLogger` class:

```python
# observability/logging/agent.py - EXISTING (719 lines)
# Location: /home/dshooter/projects/Claude_code_Quantconnect_trading_bot/observability/logging/agent.py

from observability.logging.agent import AgentLogger

# Create stream-aware logger (uses existing AgentLogger)
logger = AgentLogger(
    agent_id="exec-agent-1",
    session_id="PUCS-20251207-001"
)

# Log stream assignment (new field, existing method)
logger.log(
    level=LogLevel.INFO,
    message="Agent assigned to EXEC stream",
    category=LogCategory.AGENT,
    data={"stream_id": "EXEC", "task": "Fill predictor enhancement"}
)

# File operations already supported
logger.log_file_modified("execution/fill_predictor.py", "Added ML model")

# Conflict detection already supported
conflicts = logger.check_conflicts_before_change("execution/fill_predictor.py")
```

**Existing AgentLogger Features (DO NOT DUPLICATE):**

- Session tracking (`log_session_start`, `log_session_end`, `log_session_handoff`)
- File operations (`log_file_created`, `log_file_modified`, `log_file_deleted`)
- Task tracking (`log_task_started`, `log_task_completed`, `log_task_failed`)
- Git operations (`log_git_commit`, `log_git_push`)
- Conflict checking (`check_conflicts_before_change`)
- JSONL append-only logging with fcntl locking

### 3.2 Handoff System (Unified Format)

The existing handoff format in `.claude/state/handoff.json` is extended to support multi-agent handoffs:

```json
{
  "from_agent": "exec-agent-1",
  "to_agent": "exec-agent-2",
  "timestamp": "2025-12-07T10:00:00Z",
  "session_id": "PUCS-20251207-001",
  "context": {
    "completed": [
      "Fill predictor ML model added",
      "Unit tests for fill_predictor.py"
    ],
    "pending": [
      "Integration tests needed",
      "Documentation update"
    ],
    "warnings": [
      "fill_predictor.py has increased complexity"
    ],
    "commits": ["abc1234", "def5678"],
    "branch": "feature/stream-exec-fill-predictor"
  },
  "stream_context": {
    "stream_id": "EXEC",
    "locked_files": [],
    "related_files": ["execution/fill_predictor.py", "tests/test_fill_predictor.py"]
  },
  "files_to_review": ["execution/fill_predictor.py"]
}
```

**Key Changes:**

- Added `stream_context` for PUCS-specific data
- Kept `to_agent` singular (not plural) for backward compatibility
- Added `commits` and `branch` fields from existing format

### 3.3 Circuit Breaker Integration

PUCS uses the **existing** `AgentCircuitBreaker`:

```python
# agent_orchestrator.py:224-333 - EXISTING
from .claude.hooks.agents.agent_orchestrator import AgentCircuitBreaker

# Stream-level circuit breaker
class StreamCircuitBreaker:
    """Extends AgentCircuitBreaker for stream-level failure tracking."""

    def __init__(self, stream_id: str, base_breaker: AgentCircuitBreaker):
        self.stream_id = stream_id
        self.base_breaker = base_breaker

    def record_stream_failure(self, agent_id: str, error: str) -> None:
        """Record failure and potentially open circuit for stream."""
        self.base_breaker.record_failure(f"{self.stream_id}:{agent_id}")

    def can_spawn_agent(self) -> bool:
        """Check if stream can accept new agents."""
        return not self.base_breaker.is_open(self.stream_id)
```

**Existing AgentCircuitBreaker Features (DO NOT DUPLICATE):**

- Failure tracking per agent type
- Open/closed circuit state
- Automatic recovery after cooldown
- State persistence to `.claude/state/circuit_breaker.json`

### 3.4 Tracing Integration

PUCS uses existing tracing infrastructure:

```python
# agent_orchestrator.py:1407-1633 - EXISTING Tracer
# observability/otel_tracer.py - EXISTING LLMTracer

from .claude.hooks.agents.agent_orchestrator import Tracer

# Create stream-correlated trace
tracer = Tracer()
with tracer.span("PUCS-EXEC-fill-predictor") as span:
    span.set_attribute("pucs.stream_id", "EXEC")
    span.set_attribute("pucs.agent_id", "exec-agent-1")
    span.set_attribute("pucs.task", "Fill predictor enhancement")

    # ... do work ...

    span.set_attribute("pucs.files_modified", 3)
    span.set_attribute("pucs.tests_passed", True)
```

---

## 4. New PUCS Components

These are the **only new components** that PUCS adds (everything else uses existing systems):

### 4.1 Stream Lock Manager

File-level locks for parallel work coordination:

```python
# scripts/pucs/stream_locks.py - NEW

import json
import fcntl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

@dataclass
class FileLock:
    agent_id: str
    stream_id: str
    acquired_at: str
    expires_at: str
    task: str

LOCK_FILE = Path(".claude/state/pucs_locks.json")
DEFAULT_LOCK_DURATION_MIN = 30

def acquire_lock(file_path: str, agent_id: str, stream_id: str,
                 task: str, duration_min: int = DEFAULT_LOCK_DURATION_MIN) -> bool:
    """Acquire exclusive lock on a file for parallel work coordination."""
    _ensure_lock_file()

    with open(LOCK_FILE, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            data = json.load(f)

            # Check if file is already locked
            if file_path in data["locks"]:
                lock = data["locks"][file_path]
                expires = datetime.fromisoformat(lock["expires_at"])
                if datetime.now() < expires:
                    return False  # Still locked

            # Acquire lock
            now = datetime.now()
            data["locks"][file_path] = asdict(FileLock(
                agent_id=agent_id,
                stream_id=stream_id,
                acquired_at=now.isoformat(),
                expires_at=(now + timedelta(minutes=duration_min)).isoformat(),
                task=task
            ))

            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
            return True
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def release_lock(file_path: str, agent_id: str) -> bool:
    """Release lock on a file."""
    with open(LOCK_FILE, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            data = json.load(f)
            if file_path in data["locks"]:
                if data["locks"][file_path]["agent_id"] == agent_id:
                    del data["locks"][file_path]
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
                    return True
            return False
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def check_lock(file_path: str) -> Optional[FileLock]:
    """Check if a file is locked."""
    if not LOCK_FILE.exists():
        return None

    with open(LOCK_FILE, "r") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            data = json.load(f)
            if file_path in data["locks"]:
                lock = data["locks"][file_path]
                expires = datetime.fromisoformat(lock["expires_at"])
                if datetime.now() < expires:
                    return FileLock(**lock)
            return None
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def get_stream_locks(stream_id: str) -> dict[str, FileLock]:
    """Get all locks held by a stream."""
    if not LOCK_FILE.exists():
        return {}

    with open(LOCK_FILE, "r") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            data = json.load(f)
            return {
                path: FileLock(**lock)
                for path, lock in data["locks"].items()
                if lock["stream_id"] == stream_id
            }
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def cleanup_expired_locks() -> int:
    """Remove expired locks. Returns count of locks removed."""
    if not LOCK_FILE.exists():
        return 0

    with open(LOCK_FILE, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            data = json.load(f)
            now = datetime.now()
            expired = [
                path for path, lock in data["locks"].items()
                if datetime.fromisoformat(lock["expires_at"]) < now
            ]
            for path in expired:
                del data["locks"][path]

            if expired:
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
            return len(expired)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def _ensure_lock_file():
    """Ensure lock file exists."""
    if not LOCK_FILE.exists():
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCK_FILE, "w") as f:
            json.dump({"locks": {}, "version": 2}, f)
```

### 4.2 Intent Signal System

Proactive conflict prevention:

```python
# scripts/pucs/intent_signal.py - NEW

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from observability.logging.agent import AgentLogger

@dataclass
class Intent:
    agent_id: str
    stream_id: str
    files: list[str]
    description: str
    estimated_duration_min: int
    signaled_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ConflictInfo:
    file_path: str
    conflict_type: str  # LOCKED, INTENT_OVERLAP, DEPENDENCY
    holder_agent: str
    holder_stream: str
    details: str

@dataclass
class IntentResult:
    success: bool
    conflicts: list[ConflictInfo] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

INTENT_FILE = Path(".claude/state/pucs_intents.json")

def signal_intent(agent_id: str, stream_id: str, files: list[str],
                  description: str, estimated_duration_min: int) -> IntentResult:
    """
    Signal intent to work on files. Checks for conflicts before proceeding.

    Returns IntentResult with any conflicts found.
    """
    from scripts.pucs.stream_locks import check_lock

    conflicts = []
    suggestions = []

    # Check for locked files
    for file_path in files:
        lock = check_lock(file_path)
        if lock and lock.agent_id != agent_id:
            conflicts.append(ConflictInfo(
                file_path=file_path,
                conflict_type="LOCKED",
                holder_agent=lock.agent_id,
                holder_stream=lock.stream_id,
                details=f"Locked until {lock.expires_at} for: {lock.task}"
            ))
            suggestions.append(f"Wait for {lock.agent_id} or work on different files")

    # Check for intent overlaps
    existing_intents = _get_active_intents()
    for intent in existing_intents:
        if intent.agent_id == agent_id:
            continue
        overlap = set(files) & set(intent.files)
        if overlap:
            conflicts.append(ConflictInfo(
                file_path=", ".join(overlap),
                conflict_type="INTENT_OVERLAP",
                holder_agent=intent.agent_id,
                holder_stream=intent.stream_id,
                details=f"Agent intends to work on same files: {intent.description}"
            ))
            suggestions.append(f"Coordinate with {intent.agent_id} to partition work")

    # Check for cross-stream dependencies
    stream_config = _load_stream_config()
    if stream_id in stream_config:
        deps = stream_config[stream_id].get("dependencies", [])
        for file_path in files:
            for dep_stream in deps:
                if _file_in_stream(file_path, dep_stream):
                    conflicts.append(ConflictInfo(
                        file_path=file_path,
                        conflict_type="DEPENDENCY",
                        holder_agent="N/A",
                        holder_stream=dep_stream,
                        details=f"File belongs to dependency stream {dep_stream}"
                    ))

    # Record intent if no conflicts
    if not conflicts:
        _record_intent(Intent(
            agent_id=agent_id,
            stream_id=stream_id,
            files=files,
            description=description,
            estimated_duration_min=estimated_duration_min
        ))

    return IntentResult(
        success=len(conflicts) == 0,
        conflicts=conflicts,
        suggestions=suggestions
    )

def clear_intent(agent_id: str) -> None:
    """Clear an agent's intent (called when work is done)."""
    _remove_intent(agent_id)

def _get_active_intents() -> list[Intent]:
    """Get all active intents."""
    if not INTENT_FILE.exists():
        return []
    with open(INTENT_FILE, "r") as f:
        data = json.load(f)
        return [Intent(**i) for i in data.get("intents", [])]

def _record_intent(intent: Intent) -> None:
    """Record a new intent."""
    _ensure_intent_file()
    with open(INTENT_FILE, "r+") as f:
        data = json.load(f)
        # Remove any existing intent from same agent
        data["intents"] = [i for i in data["intents"] if i["agent_id"] != intent.agent_id]
        data["intents"].append({
            "agent_id": intent.agent_id,
            "stream_id": intent.stream_id,
            "files": intent.files,
            "description": intent.description,
            "estimated_duration_min": intent.estimated_duration_min,
            "signaled_at": intent.signaled_at
        })
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

def _remove_intent(agent_id: str) -> None:
    """Remove an agent's intent."""
    if not INTENT_FILE.exists():
        return
    with open(INTENT_FILE, "r+") as f:
        data = json.load(f)
        data["intents"] = [i for i in data["intents"] if i["agent_id"] != agent_id]
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

def _ensure_intent_file():
    if not INTENT_FILE.exists():
        INTENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(INTENT_FILE, "w") as f:
            json.dump({"intents": []}, f)

def _load_stream_config() -> dict:
    """Load stream configuration."""
    config_path = Path("config/pucs_streams.yaml")
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f).get("streams", {})
    return {}

def _file_in_stream(file_path: str, stream_id: str) -> bool:
    """Check if file belongs to a stream."""
    from fnmatch import fnmatch
    config = _load_stream_config()
    if stream_id in config:
        patterns = config[stream_id].get("files_pattern", "").split(",")
        return any(fnmatch(file_path, p.strip()) for p in patterns)
    return False
```

### 4.3 PUCS CLI

Command-line interface for PUCS operations:

```python
# scripts/pucs/cli.py - NEW

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="PUCS - Parallel Upgrade Coordination System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize PUCS session")
    init_parser.add_argument("--session", help="Session ID", default=None)

    # status
    status_parser = subparsers.add_parser("status", help="Show PUCS status")

    # lock
    lock_parser = subparsers.add_parser("lock", help="Acquire file lock")
    lock_parser.add_argument("file", help="File to lock")
    lock_parser.add_argument("--agent", required=True, help="Agent ID")
    lock_parser.add_argument("--stream", required=True, help="Stream ID")
    lock_parser.add_argument("--task", required=True, help="Task description")
    lock_parser.add_argument("--duration", type=int, default=30, help="Lock duration in minutes")

    # unlock
    unlock_parser = subparsers.add_parser("unlock", help="Release file lock")
    unlock_parser.add_argument("file", help="File to unlock")
    unlock_parser.add_argument("--agent", required=True, help="Agent ID")

    # intent
    intent_parser = subparsers.add_parser("intent", help="Signal work intent")
    intent_parser.add_argument("--agent", required=True, help="Agent ID")
    intent_parser.add_argument("--stream", required=True, help="Stream ID")
    intent_parser.add_argument("--files", required=True, help="Comma-separated file list")
    intent_parser.add_argument("--description", required=True, help="Work description")
    intent_parser.add_argument("--duration", type=int, default=30, help="Estimated duration")

    # cleanup
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup expired locks")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "lock":
        cmd_lock(args)
    elif args.command == "unlock":
        cmd_unlock(args)
    elif args.command == "intent":
        cmd_intent(args)
    elif args.command == "cleanup":
        cmd_cleanup(args)
    else:
        parser.print_help()

def cmd_init(args):
    """Initialize PUCS session."""
    from scripts.pucs.stream_locks import _ensure_lock_file
    from scripts.pucs.intent_signal import _ensure_intent_file

    session_id = args.session or f"PUCS-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Create state files
    _ensure_lock_file()
    _ensure_intent_file()

    # Initialize session state (using existing OvernightStateManager pattern)
    session_file = Path(".claude/state/pucs_session.json")
    session_file.parent.mkdir(parents=True, exist_ok=True)
    with open(session_file, "w") as f:
        json.dump({
            "session_id": session_id,
            "started_at": datetime.now().isoformat(),
            "streams": {},
            "agents": []
        }, f, indent=2)

    print(f"PUCS session initialized: {session_id}")
    print(f"State files created in .claude/state/")

def cmd_status(args):
    """Show PUCS status."""
    from scripts.pucs.stream_locks import LOCK_FILE
    from scripts.pucs.intent_signal import INTENT_FILE

    print("=" * 60)
    print("PUCS STATUS")
    print("=" * 60)

    # Session info
    session_file = Path(".claude/state/pucs_session.json")
    if session_file.exists():
        with open(session_file) as f:
            session = json.load(f)
        print(f"\nSession: {session['session_id']}")
        print(f"Started: {session['started_at']}")
    else:
        print("\nNo active session. Run 'pucs init' to start.")
        return

    # Locks
    print("\n--- Active Locks ---")
    if LOCK_FILE.exists():
        with open(LOCK_FILE) as f:
            locks = json.load(f).get("locks", {})
        if locks:
            for path, lock in locks.items():
                print(f"  {path}")
                print(f"    Agent: {lock['agent_id']} ({lock['stream_id']})")
                print(f"    Task: {lock['task']}")
                print(f"    Expires: {lock['expires_at']}")
        else:
            print("  (none)")
    else:
        print("  (none)")

    # Intents
    print("\n--- Active Intents ---")
    if INTENT_FILE.exists():
        with open(INTENT_FILE) as f:
            intents = json.load(f).get("intents", [])
        if intents:
            for intent in intents:
                print(f"  {intent['agent_id']} ({intent['stream_id']})")
                print(f"    Files: {', '.join(intent['files'][:3])}...")
                print(f"    Task: {intent['description']}")
        else:
            print("  (none)")
    else:
        print("  (none)")

def cmd_lock(args):
    """Acquire file lock."""
    from scripts.pucs.stream_locks import acquire_lock, check_lock

    # Check existing lock
    existing = check_lock(args.file)
    if existing:
        print(f"ERROR: File already locked by {existing.agent_id}")
        print(f"  Stream: {existing.stream_id}")
        print(f"  Task: {existing.task}")
        print(f"  Expires: {existing.expires_at}")
        sys.exit(1)

    success = acquire_lock(
        file_path=args.file,
        agent_id=args.agent,
        stream_id=args.stream,
        task=args.task,
        duration_min=args.duration
    )

    if success:
        print(f"Lock acquired: {args.file}")
    else:
        print(f"Failed to acquire lock: {args.file}")
        sys.exit(1)

def cmd_unlock(args):
    """Release file lock."""
    from scripts.pucs.stream_locks import release_lock

    success = release_lock(args.file, args.agent)
    if success:
        print(f"Lock released: {args.file}")
    else:
        print(f"Failed to release lock (not owner or not locked)")
        sys.exit(1)

def cmd_intent(args):
    """Signal work intent."""
    from scripts.pucs.intent_signal import signal_intent

    files = [f.strip() for f in args.files.split(",")]
    result = signal_intent(
        agent_id=args.agent,
        stream_id=args.stream,
        files=files,
        description=args.description,
        estimated_duration_min=args.duration
    )

    if result.success:
        print(f"Intent signaled successfully for {len(files)} files")
    else:
        print("CONFLICTS DETECTED:")
        for conflict in result.conflicts:
            print(f"  - {conflict.conflict_type}: {conflict.file_path}")
            print(f"    Holder: {conflict.holder_agent} ({conflict.holder_stream})")
            print(f"    Details: {conflict.details}")
        if result.suggestions:
            print("\nSuggestions:")
            for suggestion in result.suggestions:
                print(f"  - {suggestion}")
        sys.exit(1)

def cmd_cleanup(args):
    """Cleanup expired locks."""
    from scripts.pucs.stream_locks import cleanup_expired_locks

    count = cleanup_expired_locks()
    print(f"Cleaned up {count} expired locks")

if __name__ == "__main__":
    main()
```

---

## 5. Git Worktree Strategy

### 5.1 Setup

Each stream gets its own worktree for true isolation:

```bash
# Create worktrees for parallel streams (from main repo)
git worktree add ../trading-bot-stream-exec feature/stream-EXEC
git worktree add ../trading-bot-stream-llm feature/stream-LLM
git worktree add ../trading-bot-stream-scan feature/stream-SCAN
git worktree add ../trading-bot-stream-risk feature/stream-RISK
git worktree add ../trading-bot-stream-back feature/stream-BACK
git worktree add ../trading-bot-stream-obs feature/stream-OBS
git worktree add ../trading-bot-stream-api feature/stream-API
```

### 5.2 Commit Protocol

```bash
# Format: [PUCS-<stream>] <type>: <description>
[PUCS-EXEC] feat: Add ML model to fill predictor
[PUCS-LLM] fix: Correct sentiment analysis threshold
[PUCS-RISK] refactor: Simplify circuit breaker logic
```

### 5.3 Merge Order (Dependency-Based)

```text
Phase 1 (Parallel - No Dependencies):
  └── Streams: OBS, SCAN, BACK

Phase 2 (After Phase 1):
  └── Streams: EXEC, LLM, RISK

Phase 3 (After Phase 2):
  └── Streams: API

Phase 4 (After All):
  └── Applications: algorithms, ui, mcp
```

---

## 6. Agent Lifecycle

### 6.1 Protocol (Using Existing Systems)

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                         PUCS AGENT LIFECYCLE                              │
│                    (Integrates with agent_orchestrator.py)                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  1. REGISTER                                                              │
│     └── Use existing agent_orchestrator.py registration                  │
│     └── AgentLogger.log_session_start()                                  │
│                                                                           │
│  2. CLAIM_STREAM                                                          │
│     └── Request assignment via PUCS CLI: pucs intent                     │
│                                                                           │
│  3. SIGNAL_INTENT                                                         │
│     └── signal_intent() checks conflicts proactively                     │
│     └── AgentLogger.log("intent_signaled", ...)                          │
│                                                                           │
│  4. ACQUIRE_LOCKS                                                         │
│     └── acquire_lock() for each file                                     │
│     └── AgentLogger.log_file_modified() when starting                    │
│                                                                           │
│  5. WORK                                                                  │
│     └── Perform development tasks                                        │
│     └── Use existing Tracer for execution tracing                        │
│     └── Log all actions via AgentLogger                                  │
│                                                                           │
│  6. COMMIT                                                                │
│     └── AgentLogger.log_git_commit()                                     │
│     └── Use [PUCS-<stream>] commit prefix                                │
│                                                                           │
│  7. RELEASE_LOCKS                                                         │
│     └── release_lock() for each file                                     │
│     └── clear_intent() to remove intent                                  │
│                                                                           │
│  8. REPORT                                                                │
│     └── AgentLogger.log_task_completed()                                 │
│                                                                           │
│  9. HANDOFF                                                               │
│     └── AgentLogger.log_session_handoff()                                │
│     └── Update .claude/state/handoff.json                                │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. State Files Summary

### 7.1 Existing Files (DO NOT DUPLICATE)

| File | System | Purpose |
|------|--------|---------|
| `.claude/state/handoff.json` | AgentLogger | Agent handoff context |
| `.claude/state/agent_activity.jsonl` | AgentLogger | Activity audit trail |
| `.claude/state/ric.json` | RIC Loop | Session state |
| `.claude/state/circuit_breaker.json` | AgentCircuitBreaker | Failure tracking |
| `logs/overnight_state.json` | OvernightStateManager | Unified session state |

### 7.2 New PUCS Files

| File | System | Purpose |
|------|--------|---------|
| `.claude/state/pucs_locks.json` | Stream Lock Manager | File-level locks |
| `.claude/state/pucs_intents.json` | Intent Signal | Work intent tracking |
| `.claude/state/pucs_session.json` | PUCS CLI | Session metadata |
| `config/pucs_streams.yaml` | Stream Config | Stream definitions |

---

## 8. Conflict Resolution

### 8.1 Resolution Strategies

| Strategy | When Used | Automation |
|----------|-----------|------------|
| QUEUE | File locked by another agent | Auto: wait for lock expiry |
| PARTITION | Multiple agents need same module | Semi-auto: suggest function split |
| MERGE | Non-overlapping changes to same file | Auto: git merge (if clean) |
| ARBITRATE | Semantic conflict | Manual: coordinator decides |
| ROLLBACK | Breaking change detected | Semi-auto: revert + notify |

### 8.2 Automatic Resolution

```python
# Integrated with existing FallbackRouter pattern
from .claude.hooks.agents.agent_orchestrator import FallbackRouter

class PUCSConflictResolver:
    def __init__(self, fallback_router: FallbackRouter):
        self.fallback = fallback_router

    def resolve(self, conflict: ConflictInfo) -> str:
        if conflict.conflict_type == "LOCKED":
            # Queue: wait for lock to expire
            return f"QUEUE: Wait for {conflict.holder_agent}'s lock to expire"

        elif conflict.conflict_type == "INTENT_OVERLAP":
            # Try partition
            return f"PARTITION: Coordinate with {conflict.holder_agent}"

        elif conflict.conflict_type == "DEPENDENCY":
            # Use fallback router
            return self.fallback.get_fallback_suggestion(conflict.holder_stream)

        return "MANUAL: Requires coordinator intervention"
```

---

## 9. Quick Reference

### 9.1 CLI Commands

```bash
# Initialize session
python -m scripts.pucs.cli init --session "PUCS-20251207-001"

# Check status
python -m scripts.pucs.cli status

# Signal intent
python -m scripts.pucs.cli intent \
  --agent "exec-agent-1" \
  --stream "EXEC" \
  --files "execution/fill_predictor.py,execution/fill_ml_model.py" \
  --description "Adding ML model" \
  --duration 45

# Acquire lock
python -m scripts.pucs.cli lock execution/fill_predictor.py \
  --agent "exec-agent-1" \
  --stream "EXEC" \
  --task "ML model implementation"

# Release lock
python -m scripts.pucs.cli unlock execution/fill_predictor.py \
  --agent "exec-agent-1"

# Cleanup expired locks
python -m scripts.pucs.cli cleanup
```

### 9.2 Integration with Existing CLI

PUCS integrates with the existing agent orchestrator CLI:

```bash
# Existing: agent_orchestrator.py commands
python .claude/hooks/agents/agent_orchestrator.py status
python .claude/hooks/agents/agent_orchestrator.py ric-phase
python .claude/hooks/agents/agent_orchestrator.py trace --recent

# New: PUCS commands
python -m scripts.pucs.cli status
python -m scripts.pucs.cli intent ...
```

---

## 10. References

### Existing Systems (Integrate, Don't Duplicate)

| System | Location | Lines | Purpose |
|--------|----------|-------|---------|
| Agent Orchestrator | `.claude/hooks/agents/agent_orchestrator.py` | 2533 | Multi-agent coordination |
| AgentLogger | `observability/logging/agent.py` | 719 | Activity logging |
| OvernightStateManager | `utils/overnight_state.py` | 250+ | State persistence |
| AgentCircuitBreaker | `agent_orchestrator.py:224-333` | 109 | Failure management |
| Tracer | `agent_orchestrator.py:1407-1633` | 226 | Execution tracing |
| Multi-Agent Consensus | `llm/agents/multi_agent_consensus.py` | 701 | Trading consensus |
| Evaluation Orchestrator | `evaluation/orchestration_pipeline.py` | 1230 | Evaluation pipeline |

### External References

- [Microsoft AI Agent Design Patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
- [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Git Worktrees for Parallel Development](https://medium.com/@ooi_yee_fei/parallel-ai-development-with-git-worktrees-f2524afc3e33)
- [Claude Flow Framework](https://github.com/ruvnet/claude-flow)

---

## Appendix A: Migration from v1.1

### Files Removed (Superseded by Existing Systems)

| v1.1 Proposed | Now Uses |
|---------------|----------|
| `scripts/cross_logger.py` | `AgentLogger` |
| `scripts/agent_lock.py` | `scripts/pucs/stream_locks.py` |
| `.claude/state/orchestrator_state.json` | `OvernightStateManager` |
| `.claude/state/handoffs.json` (plural) | `.claude/state/handoff.json` (singular) |

### New Files Added

| File | Purpose |
|------|---------|
| `scripts/pucs/__init__.py` | Package init |
| `scripts/pucs/stream_locks.py` | File locking for parallel work |
| `scripts/pucs/intent_signal.py` | Proactive conflict prevention |
| `scripts/pucs/cli.py` | Command-line interface |
| `config/pucs_streams.yaml` | Stream configuration |

---

**Document Status:** Active
**Version:** 2.0 (Unified Framework)
**Last Updated:** December 7, 2025
