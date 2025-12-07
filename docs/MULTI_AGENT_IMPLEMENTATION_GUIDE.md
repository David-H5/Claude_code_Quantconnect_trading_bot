# Multi-Agent Parallel Development: Implementation Guide

**Document Type:** Implementation Guide
**Created:** 2025-12-06
**Related Research:** [Multi-Agent Parallel Bugfixing Best Practices](./research/multi_agent_parallel_bugfixing_best_practices.md)

## Overview

This guide provides **concrete implementation patterns** for coordinating multiple AI agents working on this codebase in parallel. It translates the research findings into actionable code and workflows specific to the QuantConnect Trading Bot project.

---

## Quick Start: 3-Agent Parallel Development

### Scenario: Fix 3 bugs simultaneously

```bash
# Terminal 1: Agent 1 - Fix circuit breaker bug
cd /home/dshooter/projects/Claude_code_Quantconnect_trading_bot
git checkout main && git pull
git checkout -b agent-1/bugfix-circuit-breaker-reset
# Work on models/circuit_breaker.py
pytest tests/test_circuit_breaker.py -v
git add models/circuit_breaker.py tests/test_circuit_breaker.py
git commit -m "[Agent-1][Bugfix] Fix circuit breaker reset logic"
git push origin agent-1/bugfix-circuit-breaker-reset
# Create PR via gh CLI
gh pr create --title "[Agent-1] Fix circuit breaker reset" --body "Fixes #123"

# Terminal 2: Agent 2 - Fix options scanner bug
cd /home/dshooter/projects/Claude_code_Quantconnect_trading_bot
git checkout main && git pull
git checkout -b agent-2/bugfix-options-scanner-iv
# Work on scanners/options_scanner.py
pytest tests/test_options_scanner.py -v
git add scanners/options_scanner.py tests/test_options_scanner.py
git commit -m "[Agent-2][Bugfix] Fix IV calculation in options scanner"
git push origin agent-2/bugfix-options-scanner-iv
gh pr create --title "[Agent-2] Fix IV calculation" --body "Fixes #124"

# Terminal 3: Agent 3 - Fix profit taking bug
cd /home/dshooter/projects/Claude_code_Quantconnect_trading_bot
git checkout main && git pull
git checkout -b agent-3/bugfix-profit-taking-threshold
# Work on execution/profit_taking.py
pytest tests/test_profit_taking.py -v
git add execution/profit_taking.py tests/test_profit_taking.py
git commit -m "[Agent-3][Bugfix] Fix profit threshold calculation"
git push origin agent-3/bugfix-profit-taking-threshold
gh pr create --title "[Agent-3] Fix profit threshold" --body "Fixes #125"
```

**Why This Works:**
- Each agent works on different modules (no file conflicts)
- All start from latest `main` (no merge conflicts)
- Tests run before commit (quality gate)
- PRs created immediately (visibility)

---

## File Ownership Matrix

To prevent merge conflicts, assign modules to specific agents:

| Module | Primary Agent | Files | Notes |
|--------|---------------|-------|-------|
| Circuit Breaker | Agent 1 | `models/circuit_breaker.py`, `tests/test_circuit_breaker.py` | Core safety system |
| Risk Manager | Agent 1 | `models/risk_manager.py`, `tests/test_risk_manager.py` | Related to circuit breaker |
| Options Scanner | Agent 2 | `scanners/options_scanner.py`, `tests/test_options_scanner.py` | Market scanning |
| Movement Scanner | Agent 2 | `scanners/movement_scanner.py`, `tests/test_movement_scanner.py` | Related to options |
| Profit Taking | Agent 3 | `execution/profit_taking.py`, `tests/test_profit_taking.py` | Execution logic |
| Smart Execution | Agent 3 | `execution/smart_execution.py`, `tests/test_smart_execution.py` | Related to profit |
| LLM Agents | Agent 4 | `llm/agents/*.py`, `tests/test_llm_agents.py` | AI integration |
| Indicators | Agent 5 | `indicators/*.py`, `tests/test_indicators.py` | Technical analysis |
| Algorithm | **LOCKED** | `algorithms/*.py` | Human-only, too critical |
| Config | **LOCKED** | `config/settings.json` | Shared, serialize changes |
| Shared Test | **LOCKED** | `tests/conftest.py` | Causes conflicts |

**LOCKED Files:** Only one agent (or human) can modify at a time. Use locking mechanism:

```python
# .claude/state/file_locks.json
{
  "config/settings.json": {
    "locked_by": "agent-4",
    "locked_at": "2025-12-06T14:30:00Z",
    "reason": "Adding new risk parameter"
  }
}
```

---

## Coordination Patterns

### Pattern 1: Independent Parallel (Recommended for Bug Fixes)

**Use Case:** Fixing multiple independent bugs

**Structure:**
```
Task Queue: [BUG-1, BUG-2, BUG-3, ..., BUG-N]
Agent Pool: [Agent-1, Agent-2, Agent-3]

Each agent:
1. Claim next available bug
2. Check file ownership (no conflicts?)
3. Create branch: agent-{id}/bugfix-{bug-id}
4. Fix, test, commit, PR
5. Return to pool, claim next
```

**Implementation:**

```python
# .claude/hooks/agents/task_coordinator.py
import json
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Task:
    id: str
    type: str  # "bugfix", "feature", "refactor"
    description: str
    affected_files: List[str]
    dependencies: List[str]  # Task IDs that must complete first
    priority: int  # 0=P0, 1=P1, 2=P2
    status: str  # "pending", "in_progress", "completed"
    assigned_to: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class TaskCoordinator:
    def __init__(self, state_file: Path = Path(".claude/state/task_queue.json")):
        self.state_file = state_file
        self.tasks: List[Task] = []
        self.load_state()

    def load_state(self):
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.tasks = [Task(**t) for t in data.get("tasks", [])]

    def save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tasks": [
                {
                    "id": t.id,
                    "type": t.type,
                    "description": t.description,
                    "affected_files": t.affected_files,
                    "dependencies": t.dependencies,
                    "priority": t.priority,
                    "status": t.status,
                    "assigned_to": t.assigned_to,
                    "started_at": t.started_at,
                    "completed_at": t.completed_at,
                }
                for t in self.tasks
            ]
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    def add_task(self, task: Task):
        self.tasks.append(task)
        self.save_state()

    def get_next_task(self, agent_id: str) -> Optional[Task]:
        """Get next available task for agent, considering dependencies and conflicts."""
        # Get files currently being worked on
        in_flight_files = set()
        for task in self.tasks:
            if task.status == "in_progress":
                in_flight_files.update(task.affected_files)

        # Find eligible tasks
        eligible = []
        for task in self.tasks:
            if task.status != "pending":
                continue

            # Check dependencies satisfied
            deps_satisfied = all(
                any(t.id == dep_id and t.status == "completed" for t in self.tasks)
                for dep_id in task.dependencies
            )
            if not deps_satisfied:
                continue

            # Check file conflicts
            has_conflict = any(f in in_flight_files for f in task.affected_files)
            if has_conflict:
                continue

            eligible.append(task)

        if not eligible:
            return None

        # Return highest priority task
        task = min(eligible, key=lambda t: t.priority)
        task.status = "in_progress"
        task.assigned_to = agent_id
        task.started_at = datetime.now().isoformat()
        self.save_state()
        return task

    def complete_task(self, task_id: str):
        for task in self.tasks:
            if task.id == task_id:
                task.status = "completed"
                task.completed_at = datetime.now().isoformat()
                self.save_state()
                return

# Usage in agent script
coordinator = TaskCoordinator()

# Add tasks
coordinator.add_task(Task(
    id="BUG-123",
    type="bugfix",
    description="Fix circuit breaker reset",
    affected_files=["models/circuit_breaker.py", "tests/test_circuit_breaker.py"],
    dependencies=[],
    priority=0,  # P0
    status="pending"
))

# Agent claims task
task = coordinator.get_next_task("agent-1")
if task:
    print(f"Working on: {task.description}")
    # Do work...
    coordinator.complete_task(task.id)
```

### Pattern 2: Sequential with Handoff (For Dependent Tasks)

**Use Case:** Feature requiring multiple steps in order

**Structure:**
```
Agent 1: Create data model →
Agent 2: Create API endpoints (depends on model) →
Agent 3: Create tests (depends on endpoints) →
Agent 4: Create documentation (depends on tests)
```

**Implementation:**

```python
# Define dependency chain
coordinator.add_task(Task(
    id="FEAT-001-MODEL",
    type="feature",
    description="Create User model",
    affected_files=["models/user.py"],
    dependencies=[],
    priority=1,
    status="pending"
))

coordinator.add_task(Task(
    id="FEAT-001-API",
    type="feature",
    description="Create User API",
    affected_files=["api/user.py"],
    dependencies=["FEAT-001-MODEL"],  # Waits for model
    priority=1,
    status="pending"
))

coordinator.add_task(Task(
    id="FEAT-001-TESTS",
    type="feature",
    description="Test User API",
    affected_files=["tests/test_user_api.py"],
    dependencies=["FEAT-001-API"],  # Waits for API
    priority=1,
    status="pending"
))

# Agents pull in order as dependencies complete
```

### Pattern 3: Swarm Research (Parallel Reads, Serial Writes)

**Use Case:** Research multiple topics, then consolidate

**Phase 1: Parallel Research (Read-Only)**
```python
# 5 agents research in parallel (no git conflicts)
research_tasks = [
    Task(id="RESEARCH-1", description="Research QuantConnect Greeks",
         affected_files=[], dependencies=[], priority=2, status="pending"),
    Task(id="RESEARCH-2", description="Research options pricing models",
         affected_files=[], dependencies=[], priority=2, status="pending"),
    Task(id="RESEARCH-3", description="Research risk management strategies",
         affected_files=[], dependencies=[], priority=2, status="pending"),
    Task(id="RESEARCH-4", description="Research backtesting best practices",
         affected_files=[], dependencies=[], priority=2, status="pending"),
    Task(id="RESEARCH-5", description="Research Charles Schwab API limits",
         affected_files=[], dependencies=[], priority=2, status="pending"),
]

for task in research_tasks:
    coordinator.add_task(task)

# All 5 agents can work simultaneously (no file writes)
```

**Phase 2: Serial Consolidation (Writes)**
```python
# Single agent consolidates (avoid merge conflicts)
coordinator.add_task(Task(
    id="CONSOLIDATE-1",
    type="documentation",
    description="Consolidate research findings",
    affected_files=["docs/research/options_research_2025_12_06.md"],
    dependencies=["RESEARCH-1", "RESEARCH-2", "RESEARCH-3", "RESEARCH-4", "RESEARCH-5"],
    priority=1,
    status="pending"
))
```

---

## Logging and Observability

### Correlation ID Implementation

Add to all agents to track work across parallel operations:

```python
# .claude/hooks/agents/agent_logger.py
import logging
import uuid
from contextvars import ContextVar
from pathlib import Path
import json
from datetime import datetime

# Thread-safe correlation ID storage
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='N/A')
agent_id: ContextVar[str] = ContextVar('agent_id', default='unknown')

class CorrelationIDFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = correlation_id.get()
        record.agent_id = agent_id.get()
        record.timestamp_iso = datetime.now().isoformat()
        return True

class StructuredLogger:
    def __init__(self, name: str, log_file: Path):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            '%(timestamp_iso)s [%(correlation_id)s] [%(agent_id)s] %(levelname)s: %(message)s'
        ))
        console.addFilter(CorrelationIDFilter())
        self.logger.addHandler(console)

        # File handler (JSON for parsing)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)

    def log_structured(self, level: str, event: str, **kwargs):
        data = {
            "timestamp": datetime.now().isoformat(),
            "correlation_id": correlation_id.get(),
            "agent_id": agent_id.get(),
            "level": level,
            "event": event,
            **kwargs
        }
        # Write JSON to file for machine parsing
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream.write(json.dumps(data) + "\n")
                handler.flush()

        # Write human-readable to console
        self.logger.log(getattr(logging, level.upper()), f"{event}: {kwargs}")

# Usage in agent
logger = StructuredLogger("agent-1", Path(".claude/logs/agent-1.jsonl"))

# Set context at start of task
correlation_id.set(str(uuid.uuid4()))
agent_id.set("agent-1")

logger.log_structured("INFO", "task_started",
    task_id="BUG-123",
    task_description="Fix circuit breaker reset"
)

# Log state changes
logger.log_structured("INFO", "state_changed",
    task_id="BUG-123",
    previous_state="pending",
    new_state="in_progress"
)

# Log completion
logger.log_structured("INFO", "task_completed",
    task_id="BUG-123",
    duration_seconds=120,
    files_changed=2,
    tests_passed=True
)
```

### Centralized Log Aggregation

```python
# .claude/hooks/agents/log_aggregator.py
import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

class LogAggregator:
    def __init__(self, log_dir: Path = Path(".claude/logs")):
        self.log_dir = log_dir

    def get_all_logs(self) -> List[Dict]:
        """Aggregate logs from all agents."""
        all_logs = []
        for log_file in self.log_dir.glob("agent-*.jsonl"):
            with open(log_file) as f:
                for line in f:
                    all_logs.append(json.loads(line))

        # Sort by timestamp
        all_logs.sort(key=lambda x: x["timestamp"])
        return all_logs

    def get_logs_by_correlation(self, corr_id: str) -> List[Dict]:
        """Get all logs for a specific correlation ID (traces full task)."""
        return [log for log in self.get_all_logs() if log["correlation_id"] == corr_id]

    def get_active_tasks(self) -> Dict[str, Dict]:
        """Get currently active tasks across all agents."""
        logs = self.get_all_logs()
        active = {}

        for log in logs:
            if log.get("event") == "task_started":
                task_id = log.get("task_id")
                active[task_id] = {
                    "agent_id": log["agent_id"],
                    "started_at": log["timestamp"],
                    "description": log.get("task_description")
                }
            elif log.get("event") == "task_completed":
                task_id = log.get("task_id")
                if task_id in active:
                    del active[task_id]

        return active

    def detect_file_conflicts(self) -> List[Dict]:
        """Detect if multiple agents are working on same files."""
        logs = self.get_all_logs()
        active_files = defaultdict(list)

        for log in logs:
            if log.get("event") == "file_locked":
                file_path = log.get("file_path")
                active_files[file_path].append({
                    "agent_id": log["agent_id"],
                    "timestamp": log["timestamp"]
                })

        conflicts = []
        for file_path, agents in active_files.items():
            if len(agents) > 1:
                conflicts.append({
                    "file": file_path,
                    "agents": agents
                })

        return conflicts

# Usage
aggregator = LogAggregator()

# View all active tasks
active = aggregator.get_active_tasks()
print(f"Active tasks: {len(active)}")
for task_id, info in active.items():
    print(f"  {task_id}: {info['agent_id']} - {info['description']}")

# Check for conflicts
conflicts = aggregator.detect_file_conflicts()
if conflicts:
    print("WARNING: File conflicts detected!")
    for conflict in conflicts:
        print(f"  {conflict['file']}: {[a['agent_id'] for a in conflict['agents']]}")
```

---

## Merge Conflict Prevention

### Pre-Commit Hook: Conflict Detection

```python
# .claude/hooks/validation/pre_commit_conflict_check.py
import subprocess
import sys
from pathlib import Path

def check_for_conflicts():
    """Check if committing would create merge conflicts."""
    # Fetch latest main
    subprocess.run(["git", "fetch", "origin", "main"], check=True)

    # Get files changed in current branch
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
        check=True
    )
    changed_files = result.stdout.strip().split("\n")

    # Get files changed in origin/main since branch point
    result = subprocess.run(
        ["git", "diff", "--name-only", "origin/main...HEAD"],
        capture_output=True,
        text=True,
        check=True
    )
    main_changed_files = result.stdout.strip().split("\n")

    # Find overlaps
    conflicts = set(changed_files) & set(main_changed_files)

    if conflicts:
        print("WARNING: Potential merge conflicts detected!")
        print("The following files have changed in both your branch and main:")
        for f in conflicts:
            print(f"  - {f}")
        print("\nRecommendation: Rebase on latest main before committing:")
        print("  git fetch origin && git rebase origin/main")
        return False

    return True

if __name__ == "__main__":
    if not check_for_conflicts():
        sys.exit(1)
```

### Auto-Rebase on Main Changes

```bash
# .claude/hooks/git/auto_rebase.sh
#!/bin/bash

# Run this before each commit to stay synced with main

git fetch origin main

# Check if main has new commits
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse origin/main)
BASE=$(git merge-base @ origin/main)

if [ $LOCAL = $REMOTE ]; then
    echo "Already up-to-date with main"
    exit 0
elif [ $LOCAL = $BASE ]; then
    echo "Main has new commits. Rebasing..."
    git rebase origin/main
    if [ $? -ne 0 ]; then
        echo "ERROR: Rebase failed. Resolve conflicts and run 'git rebase --continue'"
        exit 1
    fi
    echo "Rebase successful"
else
    echo "Branch has diverged from main. Manual merge required."
    exit 1
fi
```

---

## Monitoring Dashboard

### Real-Time Agent Status

```python
# .claude/hooks/agents/dashboard.py
from log_aggregator import LogAggregator
from task_coordinator import TaskCoordinator
from datetime import datetime, timedelta
import json

class AgentDashboard:
    def __init__(self):
        self.aggregator = LogAggregator()
        self.coordinator = TaskCoordinator()

    def get_status(self) -> dict:
        """Get comprehensive status of multi-agent system."""
        active_tasks = self.aggregator.get_active_tasks()
        all_tasks = self.coordinator.tasks

        # Calculate metrics
        pending_count = sum(1 for t in all_tasks if t.status == "pending")
        in_progress_count = sum(1 for t in all_tasks if t.status == "in_progress")
        completed_count = sum(1 for t in all_tasks if t.status == "completed")

        # Agent utilization
        agent_status = {}
        for task_id, info in active_tasks.items():
            agent_id = info["agent_id"]
            if agent_id not in agent_status:
                agent_status[agent_id] = []
            agent_status[agent_id].append(task_id)

        # Detect stalled tasks (> 1 hour in progress)
        stalled = []
        for task_id, info in active_tasks.items():
            started = datetime.fromisoformat(info["started_at"])
            duration = datetime.now() - started
            if duration > timedelta(hours=1):
                stalled.append({
                    "task_id": task_id,
                    "agent_id": info["agent_id"],
                    "duration_minutes": duration.total_seconds() / 60
                })

        # File conflicts
        conflicts = self.aggregator.detect_file_conflicts()

        return {
            "timestamp": datetime.now().isoformat(),
            "tasks": {
                "pending": pending_count,
                "in_progress": in_progress_count,
                "completed": completed_count,
                "total": len(all_tasks)
            },
            "agents": {
                "active": len(agent_status),
                "idle": max(0, 5 - len(agent_status)),  # Assuming 5 total agents
                "status": agent_status
            },
            "warnings": {
                "stalled_tasks": stalled,
                "file_conflicts": conflicts
            }
        }

    def print_dashboard(self):
        """Print human-readable dashboard."""
        status = self.get_status()

        print("=" * 60)
        print(f"Multi-Agent System Status - {status['timestamp']}")
        print("=" * 60)

        print("\nTasks:")
        print(f"  Pending:     {status['tasks']['pending']}")
        print(f"  In Progress: {status['tasks']['in_progress']}")
        print(f"  Completed:   {status['tasks']['completed']}")
        print(f"  Total:       {status['tasks']['total']}")

        print("\nAgents:")
        print(f"  Active: {status['agents']['active']}")
        print(f"  Idle:   {status['agents']['idle']}")
        for agent_id, tasks in status['agents']['status'].items():
            print(f"    {agent_id}: {', '.join(tasks)}")

        if status['warnings']['stalled_tasks']:
            print("\n⚠ WARNING: Stalled Tasks Detected!")
            for task in status['warnings']['stalled_tasks']:
                print(f"  {task['task_id']} ({task['agent_id']}): {task['duration_minutes']:.1f} min")

        if status['warnings']['file_conflicts']:
            print("\n⚠ WARNING: File Conflicts Detected!")
            for conflict in status['warnings']['file_conflicts']:
                print(f"  {conflict['file']}: {[a['agent_id'] for a in conflict['agents']]}")

        print("=" * 60)

# Usage
dashboard = AgentDashboard()
dashboard.print_dashboard()

# Or get JSON for external monitoring
status_json = json.dumps(dashboard.get_status(), indent=2)
Path(".claude/state/agent_status.json").write_text(status_json)
```

---

## Recommended Workflow

### Step-by-Step: 5 Agents Fixing 20 Bugs

**Setup (5 minutes):**

```bash
# 1. Initialize task queue
python3 .claude/hooks/agents/task_coordinator.py init

# 2. Load bugs from GitHub
gh issue list --label bug --json number,title,body | \
  python3 .claude/hooks/agents/import_github_issues.py

# 3. Start monitoring
python3 .claude/hooks/agents/dashboard.py --watch
```

**Execution (parallel):**

```bash
# Terminal 1-5: Each agent runs
export AGENT_ID=agent-1  # Unique per terminal
python3 .claude/hooks/agents/agent_worker.py

# Agent worker logic (simplified):
# while True:
#   task = coordinator.get_next_task(AGENT_ID)
#   if not task: break
#   create_branch(task)
#   fix_bug(task)
#   run_tests(task)
#   commit_and_push(task)
#   create_pr(task)
#   coordinator.complete_task(task.id)
```

**Monitoring (separate terminal):**

```bash
# Watch dashboard in real-time
watch -n 5 python3 .claude/hooks/agents/dashboard.py
```

**Merge (automated):**

```bash
# GitHub Actions workflow auto-merges PRs when:
# 1. All tests pass
# 2. No merge conflicts
# 3. Code coverage maintained
# 4. Security scan passes
```

---

## Integration with Existing Codebase

### Update .claude/registry.json

```json
{
  "hooks": {
    "pre_commit": [
      ".claude/hooks/validation/pre_commit_conflict_check.py"
    ],
    "task_assignment": [
      ".claude/hooks/agents/task_coordinator.py"
    ]
  },
  "agents": {
    "agent-1": {
      "specialization": "safety_systems",
      "modules": ["models/circuit_breaker.py", "models/risk_manager.py"]
    },
    "agent-2": {
      "specialization": "market_scanning",
      "modules": ["scanners/options_scanner.py", "scanners/movement_scanner.py"]
    },
    "agent-3": {
      "specialization": "execution",
      "modules": ["execution/profit_taking.py", "execution/smart_execution.py"]
    },
    "agent-4": {
      "specialization": "llm_integration",
      "modules": ["llm/agents/*.py"]
    },
    "agent-5": {
      "specialization": "testing",
      "modules": ["tests/*.py"]
    }
  }
}
```

### Update CLAUDE.md

```markdown
## Multi-Agent Parallel Development

When multiple agents work on this codebase simultaneously:

1. **Check task queue**: `python3 .claude/hooks/agents/task_coordinator.py status`
2. **Claim task**: `python3 .claude/hooks/agents/task_coordinator.py claim <agent-id>`
3. **Follow file ownership**: See `.claude/registry.json` for module assignments
4. **Log with correlation IDs**: Use `agent_logger.py` for all logging
5. **Rebase frequently**: Run `.claude/hooks/git/auto_rebase.sh` before each commit
6. **Monitor dashboard**: `python3 .claude/hooks/agents/dashboard.py`

### File Ownership Rules

- **LOCKED** files (one agent at a time): `config/settings.json`, `tests/conftest.py`, `algorithms/*.py`
- **Module ownership**: See registry.json
- **Shared files**: Require explicit coordination
```

---

## Performance Metrics

### Key Metrics to Track

```python
# .claude/hooks/agents/metrics.py
from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta

@dataclass
class ParallelEfficiencyMetrics:
    total_tasks: int
    completed_tasks: int
    total_wall_time_hours: float
    total_agent_time_hours: float
    num_agents: int
    merge_conflicts: int
    average_task_duration_minutes: float

    @property
    def theoretical_speedup(self) -> float:
        """If perfectly parallel, speedup = num_agents."""
        return self.num_agents

    @property
    def actual_speedup(self) -> float:
        """Actual speedup achieved."""
        sequential_time = self.total_agent_time_hours
        parallel_time = self.total_wall_time_hours
        return sequential_time / parallel_time if parallel_time > 0 else 0

    @property
    def parallel_efficiency(self) -> float:
        """Efficiency = actual_speedup / theoretical_speedup."""
        return self.actual_speedup / self.theoretical_speedup if self.theoretical_speedup > 0 else 0

    @property
    def conflict_rate(self) -> float:
        """Merge conflicts per task."""
        return self.merge_conflicts / self.completed_tasks if self.completed_tasks > 0 else 0

    def report(self):
        print(f"Parallel Efficiency Report")
        print(f"=" * 40)
        print(f"Tasks Completed: {self.completed_tasks}/{self.total_tasks}")
        print(f"Wall Time: {self.total_wall_time_hours:.1f} hours")
        print(f"Agent Time: {self.total_agent_time_hours:.1f} hours")
        print(f"Agents: {self.num_agents}")
        print(f"Theoretical Speedup: {self.theoretical_speedup:.1f}x")
        print(f"Actual Speedup: {self.actual_speedup:.1f}x")
        print(f"Efficiency: {self.parallel_efficiency:.1%}")
        print(f"Avg Task Duration: {self.average_task_duration_minutes:.1f} min")
        print(f"Merge Conflicts: {self.merge_conflicts} ({self.conflict_rate:.2f} per task)")

        if self.parallel_efficiency < 0.7:
            print(f"\n⚠ Low efficiency detected (<70%). Consider:")
            print(f"  - Reducing number of agents")
            print(f"  - Improving task decomposition")
            print(f"  - Optimizing coordination overhead")

# Usage
metrics = ParallelEfficiencyMetrics(
    total_tasks=20,
    completed_tasks=20,
    total_wall_time_hours=2.5,
    total_agent_time_hours=10.0,
    num_agents=5,
    merge_conflicts=3,
    average_task_duration_minutes=30
)
metrics.report()
```

---

## Troubleshooting

### Issue: High Merge Conflict Rate (>15%)

**Symptoms:**
- Many PRs failing to auto-merge
- Agents frequently rebasing

**Diagnosis:**
```bash
python3 .claude/hooks/agents/analyze_conflicts.py
```

**Solutions:**
1. Review file ownership matrix (are agents overlapping?)
2. Reduce number of parallel agents
3. Increase rebase frequency
4. Use feature flags instead of branches

### Issue: Low Parallel Efficiency (<70%)

**Symptoms:**
- Agents idle frequently
- Wall time not improving with more agents

**Diagnosis:**
```bash
python3 .claude/hooks/agents/metrics.py --analyze-idle-time
```

**Solutions:**
1. Improve task decomposition (reduce dependencies)
2. Optimize critical path
3. Reduce coordination overhead
4. Check for hidden dependencies

### Issue: Stalled Tasks (>1 hour)

**Symptoms:**
- Tasks stuck "in_progress" for extended periods

**Diagnosis:**
```bash
python3 .claude/hooks/agents/dashboard.py --show-stalled
```

**Solutions:**
1. Check agent logs for errors
2. Increase task timeout monitoring
3. Implement automatic task reassignment
4. Review task complexity (too large?)

---

## Next Steps

1. **Implement TaskCoordinator** (`.claude/hooks/agents/task_coordinator.py`)
2. **Add Structured Logging** (`.claude/hooks/agents/agent_logger.py`)
3. **Create Dashboard** (`.claude/hooks/agents/dashboard.py`)
4. **Update Registry** (`.claude/registry.json`)
5. **Test with 2 Agents** (Small-scale validation)
6. **Scale to 5 Agents** (Production use)

---

## Related Documentation

- [Multi-Agent Parallel Bugfixing Best Practices](./research/multi_agent_parallel_bugfixing_best_practices.md) - Research findings
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Codebase structure
- [CLAUDE.md](../CLAUDE.md) - Development workflow
- [RIC_CONTEXT.md](../.claude/RIC_CONTEXT.md) - RIC Loop reference
