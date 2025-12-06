#!/usr/bin/env python3
"""
Upgrade Guide Pre-loader for Overnight Sessions

Pre-loads comprehensive upgrade guides into session notes and recovery context
to ensure Claude has all necessary information even after context compaction.

Features:
- Extracts task specifications from progress files
- Generates detailed implementation guides
- Creates domain-specific context for each category
- Survives context compaction via file persistence

Usage:
    python scripts/preload_upgrade_guide.py --upgrade UPGRADE-014
    python scripts/preload_upgrade_guide.py --category "Category 4: Memory Management"
"""

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================================
# Domain Knowledge Templates
# ============================================================================

DOMAIN_KNOWLEDGE = {
    "Memory Management": {
        "patterns": [
            "Hierarchical memory (short/medium/long-term)",
            "Intelligent compaction at 80% capacity threshold",
            "Strategic context ordering to avoid lost-in-middle effect",
            "Memory TTLs with automatic purging",
            "Retrieval-augmented generation for large contexts",
        ],
        "key_concepts": [
            "Short-term: Current task context, immediate state",
            "Medium-term: Subgoal tracking, session decisions",
            "Long-term: Domain knowledge, learned patterns",
            "Compaction triggers: Token count, time-based, relevance decay",
        ],
        "implementation_hints": [
            "Use dataclasses for memory entries with timestamps",
            "Implement LRU eviction for short-term memory",
            "Use semantic similarity for relevance scoring",
            "Persist long-term memory to SQLite/JSON",
        ],
        "test_cases": [
            "Test memory overflow handling",
            "Test compaction preserves critical context",
            "Test retrieval accuracy after compaction",
            "Test TTL expiration",
        ],
    },
    "Cost Optimization": {
        "patterns": [
            "Model cascading (budget -> standard -> premium)",
            "Response caching with semantic similarity",
            "Budget controls with alerts at 60/80/100%",
            "Context pruning for 40-50% token reduction",
            "RAG optimization for 70% prompt size reduction",
        ],
        "key_concepts": [
            "Token economics: Input vs output costs",
            "Cache hit rates: Target >30% for common queries",
            "Model routing: Simple tasks to cheaper models",
            "Prompt compression: Remove redundancy, use references",
        ],
        "implementation_hints": [
            "Use embedding similarity for cache lookup (threshold 0.92+)",
            "Implement tiered model selection based on complexity score",
            "Track costs per request/agent/category",
            "Use summarization for long context inputs",
        ],
        "test_cases": [
            "Test cache hit/miss scenarios",
            "Test model routing decisions",
            "Test budget enforcement and alerts",
            "Test cost tracking accuracy",
        ],
    },
    "State Persistence": {
        "patterns": [
            "LangGraph-style checkpointing API",
            "Thread management for separate conversations",
            "PostgreSQL/SQLite checkpointer backends",
            "Checkpoint versioning and rollback",
        ],
        "key_concepts": [
            "Checkpoint = state snapshot at a point in time",
            "Thread = conversation/execution context",
            "Checkpoint tuple = (config, checkpoint, parent_config)",
            "Pending writes = uncommitted changes",
        ],
        "implementation_hints": [
            "Mirror LangGraph's BaseCheckpointSaver interface",
            "Use JSON serialization for state",
            "Implement get_tuple, put, list methods",
            "Support checkpoint filtering by thread_id, step",
        ],
        "test_cases": [
            "Test checkpoint save/restore",
            "Test checkpoint listing and filtering",
            "Test persistence across sessions",
            "Test rollback functionality",
        ],
    },
    "Testing & Simulation": {
        "patterns": [
            "Simulation-based testing with mock users",
            "Sandboxed execution environments",
            "LLM-as-a-Judge evaluation",
            "Cross-environment validation",
        ],
        "key_concepts": [
            "Simulation = replay scenarios without live systems",
            "Sandbox = isolated environment with controlled state",
            "LLM Judge = use LLM to evaluate agent outputs",
            "Monte Carlo = statistical testing with random scenarios",
        ],
        "implementation_hints": [
            "Create UserSimulator class with configurable behaviors",
            "Use Docker for sandboxed execution",
            "Implement evaluation rubrics for LLM judge",
            "Generate diverse test scenarios programmatically",
        ],
        "test_cases": [
            "Test simulator generates valid scenarios",
            "Test sandbox isolation",
            "Test LLM judge consistency",
            "Test cross-environment parity",
        ],
    },
    "Self-Improvement": {
        "patterns": [
            "Feedback loop with outcome capture",
            "Automatic prompt optimization (APO)",
            "Evaluator-Optimizer pattern",
            "Performance trend analysis",
        ],
        "key_concepts": [
            "Feedback loop = observe outcome, adjust behavior",
            "APO = iteratively refine prompts based on performance",
            "Evaluator = scores agent outputs objectively",
            "Optimizer = modifies prompts to improve scores",
        ],
        "implementation_hints": [
            "Log all decisions with confidence and outcomes",
            "Calculate calibration error (confidence vs accuracy)",
            "Use A/B testing for prompt variations",
            "Implement prompt mutation operators",
        ],
        "test_cases": [
            "Test feedback capture and storage",
            "Test prompt optimization converges",
            "Test evaluator scoring consistency",
            "Test trend detection accuracy",
        ],
    },
    "Workspace Management": {
        "patterns": [
            "AGENTS.md standard for directory instructions",
            "Real-time codebase indexing",
            "Event-based triggers for agent actions",
            "Multi-agent coordination protocols",
        ],
        "key_concepts": [
            "AGENTS.md = per-directory agent instructions",
            "Codebase index = fast symbol/file lookup",
            "Event triggers = file changes, git commits, etc.",
            "Coordination = agent communication and task handoff",
        ],
        "implementation_hints": [
            "Parse AGENTS.md files recursively",
            "Use AST parsing for code indexing",
            "Implement file watcher for events",
            "Use message passing for agent coordination",
        ],
        "test_cases": [
            "Test AGENTS.md parsing and merging",
            "Test index accuracy and performance",
            "Test event detection and handling",
            "Test multi-agent task handoff",
        ],
    },
}

# ============================================================================
# Progress File Parser
# ============================================================================


def parse_progress_file(progress_path: Path) -> dict[str, Any]:
    """Parse progress file to extract categories and tasks."""
    if not progress_path.exists():
        return {"categories": [], "tasks": []}

    content = progress_path.read_text()
    categories = []
    tasks = []

    current_category = None
    current_priority = "P1"

    for line in content.split("\n"):
        # Match category headers
        cat_match = re.match(r"^##\s*CATEGORY\s+(\d+):\s*(.+?)\s*\((P[0-2])\)", line)
        if cat_match:
            cat_num = cat_match.group(1)
            cat_name = cat_match.group(2)
            current_priority = cat_match.group(3)
            current_category = {
                "number": int(cat_num),
                "name": cat_name,
                "priority": current_priority,
                "status": "pending",
                "tasks": [],
            }
            if "- COMPLETED" in line:
                current_category["status"] = "completed"
            categories.append(current_category)
            continue

        # Match tasks
        task_match = re.match(r"^-\s*\[([ x])\]\s*(.+)$", line.strip())
        if task_match and current_category:
            is_complete = task_match.group(1) == "x"
            task_desc = task_match.group(2)
            task = {
                "description": task_desc,
                "complete": is_complete,
                "category": f"Category {current_category['number']}: {current_category['name']}",
                "priority": current_priority,
            }
            current_category["tasks"].append(task)
            tasks.append(task)

    return {"categories": categories, "tasks": tasks}


# ============================================================================
# Guide Generator
# ============================================================================


def find_research_doc(category_name: str, project_root: Path = Path(".")) -> Path | None:
    """Find the research document for a category."""
    research_dir = project_root / "docs" / "research"
    if not research_dir.exists():
        return None

    # Extract category number
    import re

    cat_match = re.match(r"Category (\d+):", category_name)
    if not cat_match:
        return None

    cat_num = cat_match.group(1)

    # Look for matching research doc
    for doc in research_dir.glob(f"UPGRADE-*-CAT{cat_num}-*.md"):
        return doc

    return None


def generate_category_guide(category_name: str, tasks: list[dict], project_root: Path = Path(".")) -> str:
    """Generate comprehensive guide for a category."""
    # Find domain knowledge
    domain_key = None
    for key in DOMAIN_KNOWLEDGE:
        if key.lower() in category_name.lower():
            domain_key = key
            break

    # Find research doc
    research_doc = find_research_doc(category_name, project_root)

    guide = f"""
## Implementation Guide: {category_name}
"""
    if research_doc:
        guide += f"""
**Research Doc**: `docs/research/{research_doc.name}`
"""

    guide += """
### Tasks to Complete
"""
    for task in tasks:
        status = "✓" if task["complete"] else "○"
        guide += f"- [{status}] {task['description']}\n"

    if domain_key and domain_key in DOMAIN_KNOWLEDGE:
        knowledge = DOMAIN_KNOWLEDGE[domain_key]

        guide += """
### Design Patterns
"""
        for pattern in knowledge["patterns"]:
            guide += f"- {pattern}\n"

        guide += """
### Key Concepts
"""
        for concept in knowledge["key_concepts"]:
            guide += f"- {concept}\n"

        guide += """
### Implementation Hints
"""
        for hint in knowledge["implementation_hints"]:
            guide += f"- {hint}\n"

        guide += """
### Test Cases to Write
"""
        for test in knowledge["test_cases"]:
            guide += f"- {test}\n"

    guide += """
### Completion Criteria
- All tasks marked with [x] in progress file
- Tests pass with >70% coverage
- No linting errors
- Documentation updated
"""

    return guide


def get_ric_loop_checklist() -> str:
    """Get the RIC Loop quick reference checklist."""
    return """
## RIC Loop Quick Reference

For complex tasks, use the Meta-RIC Loop (7 phases, min 3 iterations):

| Phase | Purpose | Gate Criteria |
|-------|---------|---------------|
| **0. Research** | Online research | Sources timestamped |
| **1. Upgrade Path** | Define scope, success criteria | Clear target state |
| **2. Checklist** | Break into tasks (30min-4hr each) | Tasks prioritized P0/P1/P2 |
| **3. Coding** | Execute ONE component at a time | Tests pass, committed |
| **4. Double-Check** | Verify completeness | >70% coverage |
| **5. Introspection** | Find gaps, bugs, expansion ideas | All gaps documented |
| **6. Metacognition** | Self-reflection, classify insights | P0/P1/P2 classified |
| **7. Integration** | Loop decision: continue or exit | All P0-P2 resolved |

### Exit Rules
- Minimum 3 iterations required
- ALL P0, P1, AND P2 must be resolved
- Exit only when iteration >= 3 AND no P0/P1/P2 remain

### Slash Commands
- `/ric-start` - Start new RIC loop
- `/ric-introspect` - Run introspection phase
- `/ric-converge` - Check convergence and decide
"""


def get_recent_upgrades_summary(project_root: Path) -> str:
    """Get summary of recent upgrade documents."""
    research_dir = project_root / "docs" / "research"
    if not research_dir.exists():
        return ""

    upgrades = sorted(research_dir.glob("UPGRADE-*.md"), reverse=True)[:5]
    if not upgrades:
        return ""

    summary = "\n## Recent Upgrade Documents\n\n"
    for upgrade in upgrades:
        content = upgrade.read_text()
        # Extract title and status
        title_match = re.search(r"^#\s*(.+)$", content, re.MULTILINE)
        status_match = re.search(r"Status:\s*(\w+)", content, re.IGNORECASE)

        title = title_match.group(1) if title_match else upgrade.stem
        status = status_match.group(1) if status_match else "Unknown"

        summary += f"- **{upgrade.stem}**: {title[:60]}... [{status}]\n"

    return summary


def generate_full_upgrade_guide(
    upgrade_id: str,
    progress_data: dict[str, Any],
    project_root: Path = Path("."),
) -> str:
    """Generate comprehensive upgrade guide."""
    guide = f"""# {upgrade_id} - Comprehensive Implementation Guide

**Generated**: {datetime.now().isoformat()}
**Purpose**: Pre-loaded context for overnight autonomous sessions
**Survives**: Context compaction, session restarts, crashes

---

## Quick Start (READ FIRST)

1. **Check current state**: `tail -50 claude-progress.txt`
2. **Find next task**: Look for first unchecked `- [ ]` item
3. **Start working**: Implement task, run tests, commit
4. **Mark complete**: Change `- [ ]` to `- [x]` in progress file

### Key Files
| File | Purpose |
|------|---------|
| `claude-progress.txt` | Task list and completion status |
| `claude-session-notes.md` | Context for relay-race pattern |
| `claude-recovery-context.md` | Post-compaction recovery |
| This file | Domain knowledge and guides |

---
{get_ric_loop_checklist()}
---

## Overview

This guide contains all information needed to complete {upgrade_id}.
After context compaction, Claude should READ THIS FILE to restore context.

## Priority Order

Complete categories in this order:
1. **P0 (Critical)**: Must complete first - BLOCKING
2. **P1 (Important)**: Required for full implementation - REQUIRED
3. **P2 (Nice-to-have)**: Also REQUIRED for overnight sessions

**IMPORTANT**: The stop hook enforces 100% completion by default.
You cannot stop until ALL P0, P1, AND P2 tasks are complete.

---
"""

    # Group by priority
    by_priority: dict[str, list] = {"P0": [], "P1": [], "P2": []}
    for cat in progress_data["categories"]:
        priority = cat["priority"]
        by_priority[priority].append(cat)

    for priority in ["P0", "P1", "P2"]:
        if by_priority[priority]:
            guide += f"\n## {priority} Categories\n"
            for cat in by_priority[priority]:
                cat_name = f"Category {cat['number']}: {cat['name']}"
                status = "✓ COMPLETE" if cat["status"] == "completed" else "○ PENDING"
                guide += f"\n### {cat_name} [{status}]\n"

                if cat["status"] != "completed":
                    guide += generate_category_guide(cat_name, cat["tasks"], project_root)

    # Add recent upgrades summary
    guide += get_recent_upgrades_summary(project_root)

    guide += """
---

## Robustness Guidelines

### After Context Compaction
1. Read this file (claude-upgrade-guide.md)
2. Read claude-progress.txt for current status
3. Read claude-recovery-context.md for session state
4. Continue from where you left off

### Error Recovery
- If a task fails, document the error in BLOCKERS section
- Try alternative approaches before giving up
- Create git checkpoint after each successful task

### Verification
- Run tests after each implementation
- Check for linting errors
- Verify imports work correctly

---

## Quick Reference

### Files to Update
- `claude-progress.txt` - Mark tasks [x] when complete
- `claude-session-notes.md` - Document key decisions
- Test files in `tests/` - Create tests for new code

### Commands
- Run tests: `.venv/bin/pytest tests/ -v`
- Lint: `ruff check .`
- Checkpoint: `git add -A && git commit -m "checkpoint: [description]"`
- QA Check: `python scripts/qa_validator.py`

### Slash Commands
- `/overnight` - Setup overnight session
- `/overnight-status` - Check session status
- `/ric-start` - Start RIC loop for complex tasks
- `/qa-debug` - Run debug-focused QA checks
"""

    return guide


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Pre-load upgrade guide for overnight sessions")
    parser.add_argument("--upgrade", default="UPGRADE-014", help="Upgrade ID")
    parser.add_argument("--progress-file", default="claude-progress.txt", help="Progress file path")
    parser.add_argument("--output", default="claude-upgrade-guide.md", help="Output guide file")
    parser.add_argument("--category", help="Generate guide for specific category only")
    args = parser.parse_args()

    progress_path = Path(args.progress_file)
    progress_data = parse_progress_file(progress_path)

    if args.category:
        # Find specific category
        for cat in progress_data["categories"]:
            if args.category.lower() in f"Category {cat['number']}: {cat['name']}".lower():
                guide = generate_category_guide(
                    f"Category {cat['number']}: {cat['name']}",
                    cat["tasks"],
                    Path("."),
                )
                print(guide)
                return

        print(f"Category not found: {args.category}")
        return

    # Generate full guide
    guide = generate_full_upgrade_guide(args.upgrade, progress_data, Path("."))

    output_path = Path(args.output)
    output_path.write_text(guide)

    print(f"Upgrade guide generated: {output_path}")
    print(f"Categories: {len(progress_data['categories'])}")
    print(f"Tasks: {len(progress_data['tasks'])}")

    # Summary
    complete = sum(1 for t in progress_data["tasks"] if t["complete"])
    total = len(progress_data["tasks"])
    print(f"Progress: {complete}/{total} ({complete / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
