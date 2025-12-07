# Parallel Agent Coordination System

A comprehensive framework for coordinating multiple AI agents working on the codebase in parallel, based on industry best practices and codebase analysis.

**Created**: 2025-12-06
**Based on**: Research from Anthropic, Metacircuits, AI Native Dev, and codebase analysis

---

## Executive Summary

This document defines:
1. **Work streams** that can be executed in parallel without conflicts
2. **Coordination protocols** to prevent merge conflicts and duplicate work
3. **File locking** strategy for shared resources
4. **Cross-logging** system for agent communication
5. **Automated orchestration** for agent spawning and merging

---

## Part 1: Parallelizable Work Streams

### Module Dependency Analysis

Based on codebase analysis, modules are organized into 5 layers:

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 4: APPLICATIONS (sequential - depend on all layers)       │
│ ├─ algorithms/  ├─ api/  ├─ ui/  ├─ mcp/  ├─ evaluation/       │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 3: DOMAIN LOGIC (parallelizable with coordination)        │
│ ├─ llm/  ├─ execution/  ├─ backtesting/  ├─ scanners/          │
│ ├─ indicators/  ├─ analytics/                                   │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 2: CORE MODELS (parallelizable within layer)              │
│ ├─ models/  ├─ compliance/                                      │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 1: INFRASTRUCTURE (fully parallelizable)                  │
│ ├─ infrastructure/  ├─ observability/                           │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 0: FOUNDATION (fully parallelizable)                      │
│ ├─ config/  ├─ utils/  ├─ data/                                │
└─────────────────────────────────────────────────────────────────┘
```

### Work Stream Definitions

#### Stream A: Foundation & Infrastructure
**Parallelizable**: YES (no dependencies)
**Agents**: 2-3

| Agent ID | Scope | Files |
|----------|-------|-------|
| `agent-A1` | Config module | `config/*.py` |
| `agent-A2` | Utils module | `utils/*.py` |
| `agent-A3` | Infrastructure | `infrastructure/*.py`, `data/*.py` |

#### Stream B: Observability & Compliance
**Parallelizable**: YES (isolated monitoring)
**Agents**: 2

| Agent ID | Scope | Files |
|----------|-------|-------|
| `agent-B1` | Observability | `observability/**/*.py` |
| `agent-B2` | Compliance | `compliance/*.py` |

#### Stream C: Analytics & Indicators
**Parallelizable**: YES (read-only calculations)
**Agents**: 2

| Agent ID | Scope | Files |
|----------|-------|-------|
| `agent-C1` | Analytics | `analytics/*.py` |
| `agent-C2` | Indicators | `indicators/*.py` |

#### Stream D: Scanners & Backtesting
**Parallelizable**: YES (independent analysis)
**Agents**: 2

| Agent ID | Scope | Files |
|----------|-------|-------|
| `agent-D1` | Scanners | `scanners/*.py` |
| `agent-D2` | Backtesting | `backtesting/*.py` |

#### Stream E: Models (Coordinated)
**Parallelizable**: PARTIAL (some interdependencies)
**Agents**: 3 (with coordination)

| Agent ID | Scope | Files | Dependencies |
|----------|-------|-------|--------------|
| `agent-E1` | Risk Core | `models/risk_manager.py`, `models/circuit_breaker.py` | None |
| `agent-E2` | Analytics Models | `models/volatility_*.py`, `models/anomaly_*.py` | E1 (wait) |
| `agent-E3` | ML Models | `models/rl_*.py`, `models/attention_*.py` | None |

#### Stream F: LLM Module (Coordinated)
**Parallelizable**: PARTIAL (5 sub-streams)
**Agents**: 5 (one per sub-domain)

| Agent ID | Scope | Files |
|----------|-------|-------|
| `agent-F1` | Sentiment | `llm/sentiment*.py`, `llm/emotion_detector.py` |
| `agent-F2` | News | `llm/news*.py` |
| `agent-F3` | Decision/Reasoning | `llm/decision_logger.py`, `llm/reasoning_logger.py` |
| `agent-F4` | Providers | `llm/providers.py`, `llm/model_router.py` |
| `agent-F5` | Agents | `llm/agents/*.py` |

#### Stream G: Execution (Sequential)
**Parallelizable**: NO (high interdependency)
**Agents**: 1 (sequential)

| Agent ID | Scope | Files |
|----------|-------|-------|
| `agent-G1` | All Execution | `execution/*.py` |

#### Stream H: Applications (Sequential, After All)
**Parallelizable**: NO (depends on all layers)
**Agents**: 1 per application

| Agent ID | Scope | Files | After |
|----------|-------|-------|-------|
| `agent-H1` | Algorithms | `algorithms/*.py` | All streams |
| `agent-H2` | API | `api/*.py` | All streams |
| `agent-H3` | UI | `ui/*.py` | All streams |
| `agent-H4` | MCP | `mcp/*.py` | All streams |

---

## Part 2: Coordination Protocols

### 2.1 Git Worktree Strategy

Each parallel stream gets its own git worktree:

```bash
# Create worktrees for parallel streams
git worktree add ../trading-bot-stream-A feature/stream-A-foundation
git worktree add ../trading-bot-stream-B feature/stream-B-observability
git worktree add ../trading-bot-stream-C feature/stream-C-analytics
git worktree add ../trading-bot-stream-D feature/stream-D-scanners
git worktree add ../trading-bot-stream-E feature/stream-E-models
git worktree add ../trading-bot-stream-F feature/stream-F-llm
```

### 2.2 Branch Naming Convention

```
feature/stream-{LETTER}-{description}
├── feature/stream-A-foundation-config
├── feature/stream-A-foundation-utils
├── feature/stream-B-observability-metrics
├── feature/stream-F-llm-sentiment
└── ...
```

### 2.3 File Locking Protocol

#### Lock File Format

```json
// .claude/state/file_locks.json
{
  "locks": {
    "llm/sentiment.py": {
      "agent": "agent-F1",
      "stream": "F",
      "acquired": "2025-12-06T10:00:00Z",
      "expires": "2025-12-06T11:00:00Z",
      "task": "Consolidate sentiment to package"
    }
  },
  "version": 1
}
```

#### Lock Acquisition Protocol

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
```

---

## Part 3: Cross-Logging System

### 3.1 Agent Activity Log

All agents write to a shared activity log:

```json
// .claude/state/agent_activity.jsonl (append-only)
{"timestamp": "2025-12-06T10:00:00Z", "agent": "agent-F1", "stream": "F", "action": "started", "task": "Sentiment consolidation"}
{"timestamp": "2025-12-06T10:01:00Z", "agent": "agent-F1", "stream": "F", "action": "locked", "file": "llm/sentiment.py"}
{"timestamp": "2025-12-06T10:30:00Z", "agent": "agent-F1", "stream": "F", "action": "completed", "files_modified": 5, "tests_passed": true}
{"timestamp": "2025-12-06T10:31:00Z", "agent": "agent-F1", "stream": "F", "action": "released", "file": "llm/sentiment.py"}
```

### 3.2 Conflict Detection Log

Agents report potential conflicts:

```json
// .claude/state/conflict_warnings.jsonl
{"timestamp": "2025-12-06T10:15:00Z", "agent": "agent-F2", "warning": "news_analyzer.py imports from sentiment.py - F1 has lock", "severity": "medium"}
{"timestamp": "2025-12-06T10:20:00Z", "agent": "agent-E2", "warning": "volatility_surface.py depends on risk_manager.py - E1 in progress", "severity": "high"}
```

### 3.3 Handoff Log

For sequential dependencies:

```json
// .claude/state/handoffs.json
{
  "handoffs": [
    {
      "from_agent": "agent-E1",
      "to_agents": ["agent-E2"],
      "timestamp": "2025-12-06T11:00:00Z",
      "completed_tasks": ["Risk manager refactoring", "Circuit breaker updates"],
      "context": {
        "new_interfaces": ["RiskEnforcementChain"],
        "breaking_changes": ["RiskManager.validate() signature changed"],
        "test_status": "3548 passing"
      }
    }
  ]
}
```

---

## Part 4: Automated Orchestration

### 4.1 Orchestrator Script

```python
#!/usr/bin/env python3
"""
Agent Orchestrator - Coordinates parallel agent work streams.

Usage:
    python scripts/orchestrator.py start --streams A,B,C,D
    python scripts/orchestrator.py status
    python scripts/orchestrator.py merge --stream A
"""

import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

STREAMS = {
    "A": {"name": "Foundation", "depends_on": [], "agents": ["A1", "A2", "A3"]},
    "B": {"name": "Observability", "depends_on": [], "agents": ["B1", "B2"]},
    "C": {"name": "Analytics", "depends_on": [], "agents": ["C1", "C2"]},
    "D": {"name": "Scanners", "depends_on": [], "agents": ["D1", "D2"]},
    "E": {"name": "Models", "depends_on": ["A"], "agents": ["E1", "E2", "E3"]},
    "F": {"name": "LLM", "depends_on": ["A", "B"], "agents": ["F1", "F2", "F3", "F4", "F5"]},
    "G": {"name": "Execution", "depends_on": ["E", "F"], "agents": ["G1"]},
    "H": {"name": "Applications", "depends_on": ["A", "B", "C", "D", "E", "F", "G"], "agents": ["H1", "H2", "H3", "H4"]},
}

class Orchestrator:
    def __init__(self):
        self.state_file = Path(".claude/state/orchestrator_state.json")
        self.load_state()

    def load_state(self):
        if self.state_file.exists():
            with open(self.state_file) as f:
                self.state = json.load(f)
        else:
            self.state = {"streams": {}, "started_at": None}

    def save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def can_start_stream(self, stream_id: str) -> bool:
        """Check if all dependencies are complete."""
        deps = STREAMS[stream_id]["depends_on"]
        for dep in deps:
            if dep not in self.state["streams"]:
                return False
            if self.state["streams"][dep]["status"] != "completed":
                return False
        return True

    def start_stream(self, stream_id: str) -> bool:
        """Start agents for a stream."""
        if not self.can_start_stream(stream_id):
            print(f"Cannot start stream {stream_id}: dependencies not met")
            return False

        stream = STREAMS[stream_id]

        # Create git worktree
        worktree_path = f"../trading-bot-stream-{stream_id}"
        branch_name = f"feature/stream-{stream_id}-{stream['name'].lower()}"

        subprocess.run(["git", "worktree", "add", worktree_path, "-b", branch_name])

        self.state["streams"][stream_id] = {
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
            "worktree": worktree_path,
            "branch": branch_name,
            "agents": stream["agents"]
        }
        self.save_state()

        print(f"Started stream {stream_id} ({stream['name']}) in {worktree_path}")
        return True

    def complete_stream(self, stream_id: str):
        """Mark stream as complete and trigger dependents."""
        if stream_id in self.state["streams"]:
            self.state["streams"][stream_id]["status"] = "completed"
            self.state["streams"][stream_id]["completed_at"] = datetime.now().isoformat()
            self.save_state()

            # Check if any dependent streams can now start
            for sid, stream in STREAMS.items():
                if stream_id in stream["depends_on"]:
                    if self.can_start_stream(sid):
                        print(f"Stream {sid} ({stream['name']}) can now start")

    def merge_stream(self, stream_id: str):
        """Merge completed stream back to main."""
        if stream_id not in self.state["streams"]:
            print(f"Stream {stream_id} not found")
            return

        stream_state = self.state["streams"][stream_id]
        if stream_state["status"] != "completed":
            print(f"Stream {stream_id} not completed yet")
            return

        # Merge branch
        branch = stream_state["branch"]
        subprocess.run(["git", "checkout", "develop"])
        subprocess.run(["git", "merge", "--no-ff", branch, "-m", f"Merge stream {stream_id}: {STREAMS[stream_id]['name']}"])

        # Clean up worktree
        worktree = stream_state["worktree"]
        subprocess.run(["git", "worktree", "remove", worktree])

        print(f"Merged stream {stream_id} to develop")

def main():
    parser = argparse.ArgumentParser(description="Agent Orchestrator")
    parser.add_argument("command", choices=["start", "status", "complete", "merge"])
    parser.add_argument("--streams", help="Comma-separated stream IDs")
    parser.add_argument("--stream", help="Single stream ID")

    args = parser.parse_args()
    orch = Orchestrator()

    if args.command == "start":
        for stream_id in args.streams.split(","):
            orch.start_stream(stream_id.strip())
    elif args.command == "status":
        print(json.dumps(orch.state, indent=2))
    elif args.command == "complete":
        orch.complete_stream(args.stream)
    elif args.command == "merge":
        orch.merge_stream(args.stream)

if __name__ == "__main__":
    main()
```

### 4.2 Agent Spawn Template

Each agent is spawned with scope restrictions:

```bash
#!/bin/bash
# scripts/spawn_agent.sh

STREAM=$1
AGENT_ID=$2
WORKTREE=$3
TASK_FILE=$4

cd "$WORKTREE"

# Create agent-specific .aiignore
cat > .aiignore << EOF
# Agent $AGENT_ID can only modify files in its stream
# All other directories are read-only

# Stream A: Foundation
$([ "$STREAM" != "A" ] && echo "config/")
$([ "$STREAM" != "A" ] && echo "utils/")
$([ "$STREAM" != "A" ] && echo "data/")

# Stream B: Observability
$([ "$STREAM" != "B" ] && echo "observability/")
$([ "$STREAM" != "B" ] && echo "compliance/")

# ... (continue for all streams)
EOF

# Launch Claude Code with scope restriction
claude --task-file "$TASK_FILE" \
       --agent-id "$AGENT_ID" \
       --log-file ".claude/state/logs/$AGENT_ID.log"
```

---

## Part 5: Merge Strategy

### 5.1 Merge Order

Streams must merge in dependency order:

```
Phase 1 (Parallel): A, B, C, D
         ↓
Phase 2 (After Phase 1): E, F
         ↓
Phase 3 (After Phase 2): G
         ↓
Phase 4 (After Phase 3): H
```

### 5.2 Conflict Resolution Protocol

```python
# scripts/conflict_resolver.py

def check_conflicts(stream_a: str, stream_b: str) -> list[str]:
    """Identify potential merge conflicts between streams."""
    conflicts = []

    # Get files modified in each stream
    files_a = get_modified_files(stream_a)
    files_b = get_modified_files(stream_b)

    # Check for overlapping files
    overlap = files_a & files_b
    if overlap:
        conflicts.append(f"File overlap: {overlap}")

    # Check for import changes
    imports_a = get_new_imports(stream_a)
    imports_b = get_new_imports(stream_b)

    # Check if stream A added imports from modules stream B modified
    for imp in imports_a:
        if any(imp.startswith(f) for f in files_b):
            conflicts.append(f"Import conflict: {stream_a} imports from {imp}, modified by {stream_b}")

    return conflicts

def suggest_merge_order(streams: list[str]) -> list[str]:
    """Suggest optimal merge order based on dependency analysis."""
    # Topological sort based on imports
    pass
```

### 5.3 Post-Merge Validation

```bash
#!/bin/bash
# scripts/post_merge_validate.sh

echo "Running post-merge validation..."

# 1. Run type checking
echo "Type checking..."
mypy . --ignore-missing-imports

# 2. Run tests
echo "Running tests..."
pytest tests/ -v --tb=short

# 3. Check for import errors
echo "Checking imports..."
python -c "
import importlib
import pkgutil
import sys

packages = ['config', 'utils', 'observability', 'models', 'llm', 'execution', 'algorithms']
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f'✓ {pkg}')
    except ImportError as e:
        print(f'✗ {pkg}: {e}')
        sys.exit(1)
"

# 4. Check layer violations
echo "Checking layer violations..."
python scripts/check_layer_violations.py --strict

echo "Validation complete!"
```

---

## Part 6: Task Assignment Matrix

### Current Tasks by Stream

| Stream | Task | Priority | Agent | Status |
|--------|------|----------|-------|--------|
| **F** | Sentiment consolidation (5→1) | P0 | F1 | TODO |
| **F** | News consolidation (5→1) | P0 | F2 | TODO |
| **E** | Risk enforcement chain | P1 | E1 | TODO |
| **E** | Anomaly unification (4→1) | P1 | E2 | TODO |
| **H** | P0-1 Timestamp bug fix | P0 | H1 | TODO |
| **H** | P1-1 BaseOptionsBot class | P1 | H1 | TODO |
| **B** | Metrics consolidation | P2 | B1 | TODO |

### Parallelization Plan

```
┌─────────────────────────────────────────────────────────────────┐
│                     PARALLEL EXECUTION PLAN                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 1 (Days 1-2): Can run ALL in parallel                    │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐      │
│  │ Agent F1    │ Agent F2    │ Agent E1    │ Agent E2    │      │
│  │ Sentiment   │ News        │ Risk Chain  │ Anomaly     │      │
│  │ llm/sent*   │ llm/news*   │ models/risk │ models/anom │      │
│  └─────────────┴─────────────┴─────────────┴─────────────┘      │
│                                                                  │
│  PHASE 2 (Day 3): After Phase 1 complete                        │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ Agent H1                                            │        │
│  │ Algorithm fixes (P0-1 timestamp, P1-1 base class)   │        │
│  │ algorithms/*.py                                     │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  PHASE 3 (Day 4): Claude SDK Integration                        │
│  ┌─────────────┬─────────────┐                                  │
│  │ Agent SDK1  │ Agent SDK2  │                                  │
│  │ Subagents   │ Skills      │                                  │
│  │ .claude/ag* │ .claude/sk* │                                  │
│  └─────────────┴─────────────┘                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 7: State Files Reference

### Required State Files

```
.claude/state/
├── file_locks.json          # Current file locks
├── agent_activity.jsonl     # Activity log (append-only)
├── conflict_warnings.jsonl  # Conflict warnings (append-only)
├── handoffs.json           # Handoff context
├── orchestrator_state.json # Orchestrator state
└── logs/
    ├── agent-A1.log
    ├── agent-F1.log
    └── ...
```

### State File Initialization

```bash
#!/bin/bash
# scripts/init_parallel_state.sh

mkdir -p .claude/state/logs

# Initialize file locks
echo '{"locks": {}, "version": 1}' > .claude/state/file_locks.json

# Initialize orchestrator state
echo '{"streams": {}, "started_at": null}' > .claude/state/orchestrator_state.json

# Initialize handoffs
echo '{"handoffs": []}' > .claude/state/handoffs.json

# Create empty activity logs
touch .claude/state/agent_activity.jsonl
touch .claude/state/conflict_warnings.jsonl

echo "Parallel state initialized!"
```

---

## Part 8: Quick Start Guide

### Starting Parallel Development

```bash
# 1. Initialize state files
./scripts/init_parallel_state.sh

# 2. Start Phase 1 streams (all parallel)
python scripts/orchestrator.py start --streams F,E

# 3. In separate terminals, navigate to worktrees and start agents
# Terminal 1: Sentiment consolidation
cd ../trading-bot-stream-F
claude "Consolidate llm/sentiment*.py to llm/sentiment/ package per MASTER_CONSOLIDATION_PLAN.md"

# Terminal 2: News consolidation
cd ../trading-bot-stream-F
claude "Consolidate llm/news*.py to llm/news/ package per MASTER_CONSOLIDATION_PLAN.md"

# Terminal 3: Risk chain
cd ../trading-bot-stream-E
claude "Create RiskEnforcementChain in models/risk_chain.py per MASTER_CONSOLIDATION_PLAN.md"

# Terminal 4: Anomaly unification
cd ../trading-bot-stream-E
claude "Unify anomaly detection in models/anomaly/ per MASTER_CONSOLIDATION_PLAN.md"

# 4. When all complete, mark streams done
python scripts/orchestrator.py complete --stream F
python scripts/orchestrator.py complete --stream E

# 5. Merge in order
python scripts/orchestrator.py merge --stream F
python scripts/orchestrator.py merge --stream E

# 6. Run validation
./scripts/post_merge_validate.sh
```

---

## Sources

- [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Simon Willison: Embracing the parallel coding agent lifestyle](https://simonwillison.net/2025/Oct/5/parallel-coding-agents/)
- [AI Native Dev: Parallelizing AI Coding Agents](https://ainativedev.io/news/how-to-parallelize-ai-coding-agents)
- [Metacircuits: Managing parallel coding agents](https://metacircuits.substack.com/p/managing-parallel-coding-agents-without)
- [Medium: Parallel AI Development with Git Worktrees](https://medium.com/@ooi_yee_fei/parallel-ai-development-with-git-worktrees-f2524afc3e33)
- [Medium: Solving Parallel Workflow Conflicts](https://raminmammadzada.medium.com/solving-parallel-workflow-conflicts-between-ai-agents-and-developers-in-shared-codebases-286504422125)
- [Claude Flow Framework](https://github.com/ruvnet/claude-flow)
- [Agent-MCP Framework](https://github.com/rinadelph/Agent-MCP)
- [Curiouslychase: Running Claude Agents in Parallel](https://www.curiouslychase.com/ai-development/running-claude-agents-in-parallel-with-git-worktrees)

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [MASTER_CONSOLIDATION_PLAN.md](MASTER_CONSOLIDATION_PLAN.md) | Task definitions |
| [FIX_GUIDE.md](FIX_GUIDE.md) | Bug fixes |
| [GIT_MULTI_AGENT_WORKFLOW.md](GIT_MULTI_AGENT_WORKFLOW.md) | Git workflow |
| [MULTI_AGENT_ENHANCEMENT_PLAN.md](MULTI_AGENT_ENHANCEMENT_PLAN.md) | Claude SDK integration |
