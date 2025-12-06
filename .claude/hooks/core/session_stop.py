#!/usr/bin/env python3
"""
Enhanced Stop Hook for Robust Overnight Sessions

VERSION: 2.1

FEATURES:
1. Priority-aware task tracking (P0 > P1 > P2)
2. Minimum completion threshold enforcement (default: 100%)
3. Explicit category completion tracking
4. Compaction-proof state persistence
5. Clearer continuation prompts with specific next task
6. RIC Loop compliance checking

This hook runs when a Claude Code session ends and:
1. Checks RIC Loop state and enforces methodology
2. Checks completion status against priority requirements
3. Persists state to survive context compaction
4. Generates specific continuation prompts
5. Only allows exit when ALL required tasks are complete AND RIC criteria met

Configuration via environment variables:
- CONTINUOUS_MODE=1: Enable continuation logic
- RIC_MODE=ENFORCED|SUGGESTED|DISABLED: RIC Loop enforcement mode
- MIN_COMPLETION_PCT=100: Minimum completion percentage (default: 100)
- REQUIRE_P1=1: P1 tasks are required (default: true)
- REQUIRE_P2=1: P2 tasks are required (default: true)
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Import unified state manager
# Compute project root from current file location or use cwd as fallback
try:
    _project_root = str(Path(__file__).parent.parent.parent.parent)
except NameError:
    _project_root = str(Path.cwd())
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from utils.overnight_state import OvernightStateManager, get_overnight_state_manager


# Configuration
PROGRESS_FILE = Path("claude-progress.txt")
RIC_STATE_FILE = Path(".claude/state/ric.json")

# Unified state manager replaces fragmented session_state.json access
# Part of P1-1 integration from REMEDIATION_PLAN.md
_state_manager: OvernightStateManager | None = None

# Legacy state file for backwards compatibility migration
LEGACY_STATE_FILE = Path("logs/session_state.json")
MAX_CONTINUATION_ATTEMPTS = 20  # Increased from 10
CONTINUATION_LOG = Path("logs/continuation_history.jsonl")


def get_state_manager() -> OvernightStateManager:
    """Get or create the overnight state manager."""
    global _state_manager
    if _state_manager is None:
        _state_manager = get_overnight_state_manager()
    return _state_manager


def parse_progress_file() -> dict[str, Any]:
    """Parse progress file with priority-aware task extraction.

    Returns dict with:
    - categories: dict of category_name -> {priority, tasks, completed_count, total_count}
    - all_tasks: list of all tasks with status
    - summary: completion summary by priority
    """
    if not PROGRESS_FILE.exists():
        return {"categories": {}, "all_tasks": [], "summary": {}}

    content = PROGRESS_FILE.read_text()

    categories = {}
    all_tasks = []
    current_category = None
    current_priority = "P1"  # Default priority

    for line in content.split("\n"):
        # Detect category headers (e.g., "## CATEGORY 4: Memory Management (P1)")
        cat_match = re.match(r"^##\s*CATEGORY\s+(\d+):\s*(.+?)\s*\((P[0-2])\)", line)
        if cat_match:
            cat_num = cat_match.group(1)
            cat_name = cat_match.group(2)
            current_priority = cat_match.group(3)
            current_category = f"Category {cat_num}: {cat_name}"
            categories[current_category] = {
                "priority": current_priority,
                "tasks": [],
                "completed_count": 0,
                "total_count": 0,
                "status": "pending",
            }
            continue

        # Detect completed status header (e.g., "## CATEGORY 5: ... - COMPLETED")
        if current_category and "- COMPLETED" in line:
            categories[current_category]["status"] = "completed"
            continue

        # Detect task items (- [ ] or - [x])
        task_match = re.match(r"^-\s*\[([ x])\]\s*(.+)$", line.strip())
        if task_match and current_category:
            is_complete = task_match.group(1) == "x"
            task_desc = task_match.group(2)

            task = {
                "description": task_desc,
                "complete": is_complete,
                "category": current_category,
                "priority": categories[current_category]["priority"],
            }

            categories[current_category]["tasks"].append(task)
            categories[current_category]["total_count"] += 1
            if is_complete:
                categories[current_category]["completed_count"] += 1

            all_tasks.append(task)

    # Update category status based on tasks
    for _cat_name, cat_data in categories.items():
        if cat_data["total_count"] > 0:
            if cat_data["completed_count"] == cat_data["total_count"]:
                cat_data["status"] = "completed"
            elif cat_data["completed_count"] > 0:
                cat_data["status"] = "in_progress"

    # Calculate summary by priority
    summary = {
        "P0": {"completed": 0, "total": 0},
        "P1": {"completed": 0, "total": 0},
        "P2": {"completed": 0, "total": 0},
    }

    for task in all_tasks:
        priority = task["priority"]
        summary[priority]["total"] += 1
        if task["complete"]:
            summary[priority]["completed"] += 1

    return {"categories": categories, "all_tasks": all_tasks, "summary": summary}


def get_next_pending_task(parsed: dict[str, Any]) -> tuple[str | None, str | None]:
    """Get the next pending task with its category.

    Returns (task_description, category_name) or (None, None) if all complete.
    Prioritizes: P0 > P1 > P2
    """
    for priority in ["P0", "P1", "P2"]:
        for task in parsed["all_tasks"]:
            if task["priority"] == priority and not task["complete"]:
                return task["description"], task["category"]
    return None, None


def get_pending_categories(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Get list of categories that still have pending tasks.

    Returns list sorted by priority (P0 first), then by category number.
    """
    pending = []
    for cat_name, cat_data in parsed["categories"].items():
        if cat_data["status"] != "completed" and cat_data["total_count"] > 0:
            pending.append(
                {
                    "name": cat_name,
                    "priority": cat_data["priority"],
                    "completed": cat_data["completed_count"],
                    "total": cat_data["total_count"],
                    "pct": cat_data["completed_count"] / cat_data["total_count"] * 100
                    if cat_data["total_count"] > 0
                    else 0,
                }
            )

    # Sort by priority (P0 < P1 < P2), then by name
    pending.sort(key=lambda x: (x["priority"], x["name"]))
    return pending


def calculate_completion_status(parsed: dict[str, Any]) -> dict[str, Any]:
    """Calculate overall completion status.

    Returns dict with:
    - overall_pct: Overall completion percentage
    - p0_complete: Whether all P0 tasks are complete
    - p1_complete: Whether all P1 tasks are complete
    - p2_complete: Whether all P2 tasks are complete
    - can_exit: Whether session can exit based on requirements
    - exit_reason: Why session can/cannot exit
    """
    summary = parsed["summary"]

    # Calculate percentages
    p0_total = summary["P0"]["total"]
    p0_complete = summary["P0"]["completed"]
    p1_total = summary["P1"]["total"]
    p1_complete = summary["P1"]["completed"]
    p2_total = summary["P2"]["total"]
    p2_complete = summary["P2"]["completed"]

    total = p0_total + p1_total + p2_total
    completed = p0_complete + p1_complete + p2_complete
    overall_pct = (completed / total * 100) if total > 0 else 100

    # Check completion by priority
    p0_done = p0_complete == p0_total or p0_total == 0
    p1_done = p1_complete == p1_total or p1_total == 0
    p2_done = p2_complete == p2_total or p2_total == 0

    # Get requirements from environment
    min_completion_pct = int(os.environ.get("MIN_COMPLETION_PCT", "100"))
    require_p1 = os.environ.get("REQUIRE_P1", "1") == "1"
    require_p2 = os.environ.get("REQUIRE_P2", "1") == "1"

    # Determine if can exit
    can_exit = True
    exit_reasons = []

    if overall_pct < min_completion_pct:
        can_exit = False
        exit_reasons.append(f"Overall completion {overall_pct:.1f}% < required {min_completion_pct}%")

    if not p0_done:
        can_exit = False
        exit_reasons.append(f"P0 incomplete ({p0_complete}/{p0_total})")

    if require_p1 and not p1_done:
        can_exit = False
        exit_reasons.append(f"P1 incomplete ({p1_complete}/{p1_total})")

    if require_p2 and not p2_done:
        can_exit = False
        exit_reasons.append(f"P2 incomplete ({p2_complete}/{p2_total})")

    return {
        "overall_pct": overall_pct,
        "p0_complete": p0_done,
        "p1_complete": p1_done,
        "p2_complete": p2_done,
        "can_exit": can_exit,
        "exit_reason": "; ".join(exit_reasons) if exit_reasons else "All requirements met",
        "stats": {
            "p0": f"{p0_complete}/{p0_total}",
            "p1": f"{p1_complete}/{p1_total}",
            "p2": f"{p2_complete}/{p2_total}",
            "total": f"{completed}/{total}",
        },
    }


def load_state() -> dict[str, Any]:
    """Load persisted session state using unified OvernightStateManager.

    Note: Uses the unified state manager from utils/overnight_state.py
    which consolidates multiple state files with proper file locking.
    Part of P1-1 integration from REMEDIATION_PLAN.md.
    """
    manager = get_state_manager()
    state = manager.load()

    # Convert to dict for backwards compatibility with existing code
    return {
        "continuation_count": state.continuation_count,
        "last_continuation": state.last_updated,
        "session_start": state.started_at or datetime.now(timezone.utc).isoformat(),
        "tasks_at_start": state.total_tasks,
        "completed_since_start": state.completed_count,
        "restart_count": state.restart_count,
        "goal": state.goal,
        "session_id": state.session_id,
        "completion_pct": state.completion_pct,
    }


def check_ric_compliance() -> dict[str, Any]:
    """Check RIC Loop compliance status.

    Returns dict with:
    - active: Whether RIC session is active
    - can_exit: Whether RIC criteria allow exit
    - reason: Explanation of exit status
    - iteration: Current iteration
    - phase: Current phase
    - insights_open: Count of open insights by priority
    """
    result = {
        "active": False,
        "can_exit": True,
        "reason": "No active RIC session",
        "iteration": 0,
        "max_iterations": 5,
        "min_iterations": 3,
        "phase": 0,
        "phase_name": "",
        "insights_open": {"P0": 0, "P1": 0, "P2": 0},
    }

    # Check if RIC mode is enabled
    ric_mode = os.environ.get("RIC_MODE", "SUGGESTED").upper()
    if ric_mode == "DISABLED":
        return result

    if not RIC_STATE_FILE.exists():
        return result

    try:
        state = json.loads(RIC_STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return result

    # Check if session is active
    if not state.get("upgrade_id"):
        return result

    result["active"] = True
    result["iteration"] = state.get("current_iteration", 1)
    result["max_iterations"] = state.get("max_iterations", 5)
    result["min_iterations"] = state.get("min_iterations", 3)
    result["phase"] = state.get("current_phase", 0)

    # Phase names
    phase_names = {
        0: "Research",
        1: "Upgrade Path",
        2: "Checklist",
        3: "Coding",
        4: "Double-Check",
        5: "Introspection",
        6: "Metacognition",
        7: "Integration",
    }
    result["phase_name"] = phase_names.get(result["phase"], "Unknown")

    # Count open insights
    insights = state.get("insights", [])
    for insight in insights:
        if isinstance(insight, dict) and insight.get("status") != "resolved":
            priority = insight.get("priority", "P1")
            if priority in result["insights_open"]:
                result["insights_open"][priority] += 1

    # Apply RIC Loop exit rules
    p0_open = result["insights_open"]["P0"]
    p1_open = result["insights_open"]["P1"]
    p2_open = result["insights_open"]["P2"]

    # Rule 1: Minimum iterations
    if result["iteration"] < result["min_iterations"]:
        result["can_exit"] = False
        result["reason"] = f"RIC: Min {result['min_iterations']} iterations required (current: {result['iteration']})"
        return result

    # Rule 2: P0 insights must be resolved
    if p0_open > 0:
        result["can_exit"] = False
        result["reason"] = f"RIC: {p0_open} P0 (critical) insights still open"
        return result

    # Rule 3: P1 insights must be resolved
    if p1_open > 0:
        result["can_exit"] = False
        result["reason"] = f"RIC: {p1_open} P1 (important) insights still open"
        return result

    # Rule 4: P2 insights must be resolved (not optional!)
    if p2_open > 0:
        result["can_exit"] = False
        result["reason"] = f"RIC: {p2_open} P2 (polish) insights still open - P2 is REQUIRED"
        return result

    # Rule 5: Must be at Phase 7 (Integration) or have completed it
    if result["phase"] < 7:
        result["can_exit"] = False
        result["reason"] = f"RIC: Must complete all phases (current: Phase {result['phase']} - {result['phase_name']})"
        return result

    result["can_exit"] = True
    result["reason"] = "RIC: All exit criteria met"
    return result


def generate_ric_continuation_prompt(ric_status: dict[str, Any]) -> str:
    """Generate RIC-specific continuation prompt."""
    prompt = f"""
RIC LOOP CONTINUATION REQUIRED
==============================
Iteration: {ric_status['iteration']}/{ric_status['max_iterations']} (min: {ric_status['min_iterations']})
Phase: {ric_status['phase']} - {ric_status['phase_name']}

Open Insights:
  P0 (Critical): {ric_status['insights_open']['P0']}
  P1 (Important): {ric_status['insights_open']['P1']}
  P2 (Polish): {ric_status['insights_open']['P2']}

Exit blocked: {ric_status['reason']}

NEXT ACTIONS:
"""
    if ric_status["iteration"] < ric_status["min_iterations"]:
        prompt += f"""
1. Complete current phase ({ric_status['phase_name']})
2. Advance through remaining phases
3. Begin iteration {ric_status['iteration'] + 1}
4. Repeat until iteration >= {ric_status['min_iterations']}
"""
    elif ric_status["insights_open"]["P0"] > 0:
        prompt += """
1. Review open P0 insights in ric-progress.md
2. Address each P0 insight immediately
3. Mark as resolved with resolution description
4. Run: python3 .claude/hooks/ric_state_manager.py resolve --insight-id <ID> --resolution "<text>"
"""
    elif ric_status["insights_open"]["P1"] > 0:
        prompt += """
1. Review open P1 insights in ric-progress.md
2. Address each P1 insight
3. Mark as resolved with resolution description
"""
    elif ric_status["insights_open"]["P2"] > 0:
        prompt += """
1. Review open P2 insights in ric-progress.md
2. P2 is REQUIRED for exit (not optional!)
3. Address each P2 insight
4. Mark as resolved
"""
    else:
        prompt += f"""
1. Complete Phase {ric_status['phase']} - {ric_status['phase_name']}
2. Advance to next phase
3. Use format: [ITERATION {ric_status['iteration']}/{ric_status['max_iterations']}] === PHASE N ===
"""

    return prompt


def save_state(state: dict[str, Any]):
    """Save session state using unified OvernightStateManager.

    Note: Uses the unified state manager with proper file locking.
    Part of P1-1 integration from REMEDIATION_PLAN.md.
    """
    manager = get_state_manager()
    current = manager.load()

    # Map dict fields to OvernightState attributes
    if "continuation_count" in state:
        current.continuation_count = state["continuation_count"]
    if "last_continuation" in state:
        current.last_updated = state["last_continuation"]
    if "session_start" in state:
        current.started_at = state["session_start"]
    if "tasks_at_start" in state:
        current.total_tasks = state["tasks_at_start"]
    if "completed_since_start" in state:
        current.completed_count = state["completed_since_start"]
    if "restart_count" in state:
        current.restart_count = state["restart_count"]
    if "goal" in state:
        current.goal = state["goal"]
    if "session_id" in state:
        current.session_id = state["session_id"]
    if "completion_pct" in state:
        current.completion_pct = state["completion_pct"]

    manager.save(current)


def log_continuation(parsed: dict, status: dict, decision: str, reason: str):
    """Log continuation decision for analysis.

    Uses unified OvernightStateManager for state tracking and also
    writes to JSONL for historical analysis.
    Part of P1-1 integration from REMEDIATION_PLAN.md.
    """
    CONTINUATION_LOG.parent.mkdir(exist_ok=True)

    # Get completion info
    completion_pct = status.get("overall_pct", 0)
    pending_categories = len(get_pending_categories(parsed)) if parsed else 0

    # Log to unified state manager if continuing
    if decision == "continue":
        manager = get_state_manager()
        manager.record_continuation(
            reason=reason,
            completion_pct=completion_pct,
            pending_tasks=pending_categories,
        )

    # Also write to JSONL for historical analysis
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "reason": reason,
        "completion_pct": completion_pct,
        "p0": status.get("stats", {}).get("p0", "?"),
        "p1": status.get("stats", {}).get("p1", "?"),
        "p2": status.get("stats", {}).get("p2", "?"),
        "pending_categories": pending_categories,
    }

    with open(CONTINUATION_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def generate_continuation_prompt(parsed: dict, status: dict, state: dict) -> str:
    """Generate a specific continuation prompt with next task details.

    This prompt is designed to survive context compaction and provide
    clear direction to Claude.
    """
    next_task, next_category = get_next_pending_task(parsed)
    pending_cats = get_pending_categories(parsed)

    prompt = f"""
CONTINUATION REQUIRED - DO NOT STOP

Session Progress: {status['stats']['total']} ({status['overall_pct']:.1f}%)
- P0: {status['stats']['p0']} {'✓' if status['p0_complete'] else '❌'}
- P1: {status['stats']['p1']} {'✓' if status['p1_complete'] else '❌'}
- P2: {status['stats']['p2']} {'✓' if status['p2_complete'] else '❌'}

Exit blocked: {status['exit_reason']}

NEXT TASK: {next_task}
CATEGORY: {next_category}

Remaining categories ({len(pending_cats)}):
"""
    for cat in pending_cats[:5]:
        prompt += f"  - {cat['name']} ({cat['priority']}): {cat['completed']}/{cat['total']}\n"

    prompt += f"""
Continuation #{state['continuation_count'] + 1} of {MAX_CONTINUATION_ATTEMPTS}

ACTION REQUIRED:
1. Read claude-progress.txt for full task list
2. Start working on: {next_task}
3. Mark tasks with [x] as you complete them
4. DO NOT STOP until ALL P0, P1, and P2 tasks are complete
"""
    return prompt


def should_continue() -> tuple[bool, str, str]:
    """Check if Claude should continue instead of stopping.

    Returns:
        Tuple of (should_continue, reason, continuation_prompt)
    """
    # Check if continuous mode is enabled
    if os.environ.get("CONTINUOUS_MODE") != "1":
        return False, "Continuous mode not enabled", ""

    # Load persisted state
    state = load_state()

    # Check continuation count
    if state["continuation_count"] >= MAX_CONTINUATION_ATTEMPTS:
        log_continuation({}, {"overall_pct": 0, "stats": {}}, "exit", "Max attempts reached")
        return False, f"Max continuation attempts ({MAX_CONTINUATION_ATTEMPTS}) reached", ""

    # Check RIC Loop compliance first (highest priority)
    ric_status = check_ric_compliance()
    if ric_status["active"] and not ric_status["can_exit"]:
        # RIC Loop blocks exit
        state["continuation_count"] += 1
        state["last_continuation"] = datetime.now().isoformat()
        save_state(state)

        # Generate RIC-specific continuation prompt
        prompt = generate_ric_continuation_prompt(ric_status)

        log_continuation(
            {"categories": {}, "all_tasks": [], "summary": {}},
            {"overall_pct": 0, "stats": {"p0": "?", "p1": "?", "p2": "?", "total": "?"}},
            "continue",
            ric_status["reason"],
        )

        return True, ric_status["reason"], prompt

    # Parse progress file
    parsed = parse_progress_file()
    if not parsed["all_tasks"]:
        # No tasks in progress file - check if RIC session requires work
        if ric_status["active"]:
            prompt = generate_ric_continuation_prompt(ric_status)
            return True, "RIC session active but no progress tasks", prompt
        return False, "No tasks found in progress file", ""

    # Calculate completion status
    status = calculate_completion_status(parsed)

    # Check if can exit (both task completion AND RIC compliance required)
    if status["can_exit"]:
        # Double-check RIC compliance
        if ric_status["active"] and not ric_status["can_exit"]:
            # Tasks complete but RIC not satisfied
            state["continuation_count"] += 1
            state["last_continuation"] = datetime.now().isoformat()
            save_state(state)
            prompt = generate_ric_continuation_prompt(ric_status)
            log_continuation(parsed, status, "continue", ric_status["reason"])
            return True, ric_status["reason"], prompt

        log_continuation(parsed, status, "exit", status["exit_reason"])
        return False, status["exit_reason"], ""

    # Update state
    state["continuation_count"] += 1
    state["last_continuation"] = datetime.now().isoformat()
    save_state(state)

    # Generate continuation prompt (combine task and RIC info)
    prompt = generate_continuation_prompt(parsed, status, state)

    if ric_status["active"]:
        prompt += "\n" + generate_ric_continuation_prompt(ric_status)

    # Log continuation
    log_continuation(parsed, status, "continue", status["exit_reason"])

    return True, status["exit_reason"], prompt


def get_session_stats() -> dict:
    """Calculate session statistics."""
    stats = {
        "end_time": datetime.now().isoformat(),
        "commits_made": 0,
        "files_changed": 0,
    }

    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--since", "12 hours ago"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            stats["commits_made"] = len([line for line in result.stdout.strip().split("\n") if line])
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~10", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            stats["files_changed"] = len([f for f in result.stdout.strip().split("\n") if f])
    except Exception:
        pass

    return stats


def create_final_checkpoint():
    """Create a final git checkpoint commit.

    Note: We run pre-commit hooks (no --no-verify) because if hooks fail,
    that's a signal to investigate, not bypass. Checkpoint commits should
    still respect project quality gates.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            subprocess.run(["git", "add", "-A"], timeout=5)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            # Run without --no-verify to respect pre-commit hooks
            commit_result = subprocess.run(
                ["git", "commit", "-m", f"checkpoint: Session end at {timestamp}"],
                capture_output=True,
                text=True,
                timeout=60,  # Increased timeout for hooks
            )
            if commit_result.returncode != 0:
                # Log hook failure but don't crash
                import logging

                logging.warning(f"Checkpoint commit failed: {commit_result.stderr}")
            return commit_result.returncode == 0
    except subprocess.TimeoutExpired:
        import logging

        logging.warning("Checkpoint commit timed out (pre-commit hooks may be slow)")
        return False
    except Exception as e:
        import logging

        logging.warning(f"Checkpoint commit error: {e}")
        return False
    return False


def main():
    """Main stop hook execution."""
    # Check for continuation
    continue_session, reason, prompt = should_continue()

    if continue_session:
        state = load_state()
        print(f"\n{'=' * 60}")
        print(f"CONTINUATION MODE ({state['continuation_count']}/{MAX_CONTINUATION_ATTEMPTS})")
        print(f"{'=' * 60}")
        print(f"Reason: {reason}")
        print(prompt)

        # Output JSON block for Claude to parse
        result = {
            "decision": "block",
            "reason": reason,
            "continuation_count": state["continuation_count"],
            "prompt": prompt[:500],  # Truncate for JSON
        }
        print(json.dumps(result))

        return 2  # Exit code 2 = continue

    # Normal exit
    print("\n" + "=" * 60)
    print("SESSION ENDING - All requirements met")
    print("=" * 60)
    print(f"Exit reason: {reason}")

    # Cleanup
    stats = get_session_stats()
    create_final_checkpoint()

    # Clear state file for next session
    # Uses unified state file from OvernightStateManager
    manager = get_state_manager()
    if manager.state_file.exists():
        manager.state_file.unlink()

    # Also clean up legacy state file if it exists
    if LEGACY_STATE_FILE.exists():
        LEGACY_STATE_FILE.unlink()

    print(f"Session stats: commits={stats['commits_made']}, files={stats['files_changed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
