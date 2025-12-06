#!/usr/bin/env python3
"""
Get progress stats for bash scripts.

Part of P1-4 integration from REMEDIATION_PLAN.md

Usage in bash:
    eval $(python3 scripts/get_progress_stats.py)

This script uses the unified ProgressParser to get progress statistics
and outputs them as shell variable assignments.
"""

import sys
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.progress_parser import ProgressParser


def main():
    """Load progress and print shell variable assignments."""
    parser = ProgressParser()
    data = parser.parse()

    # Basic stats
    print(f"TOTAL_TASKS={data.total_count}")
    print(f"COMPLETED_TASKS={data.completed_count}")
    print(f"PENDING_TASKS={len(data.pending_tasks)}")
    print(f"COMPLETION_PCT={int(data.completion_pct)}")

    # Priority-specific stats
    p0_pending = len(data.get_pending_by_priority("P0"))
    p1_pending = len(data.get_pending_by_priority("P1"))
    p2_pending = len(data.get_pending_by_priority("P2"))
    print(f"P0_PENDING={p0_pending}")
    print(f"P1_PENDING={p1_pending}")
    print(f"P2_PENDING={p2_pending}")

    # Completion flags for each priority
    print(f"P0_COMPLETE={'1' if p0_pending == 0 else '0'}")
    print(f"P1_COMPLETE={'1' if p1_pending == 0 else '0'}")
    print(f"P2_COMPLETE={'1' if p2_pending == 0 else '0'}")
    print(f"ALL_COMPLETE={'1' if data.total_count > 0 and data.completion_pct >= 100 else '0'}")

    # Next task info
    next_task = data.get_next_task()
    if next_task:
        # Escape single quotes for bash
        desc = next_task.description.replace("'", "'\\''")
        print(f"NEXT_TASK='{desc}'")
        print(f"NEXT_TASK_PRIORITY='{next_task.priority}'")
        print(f"NEXT_TASK_CATEGORY='{next_task.category}'")
    else:
        print("NEXT_TASK=''")
        print("NEXT_TASK_PRIORITY=''")
        print("NEXT_TASK_CATEGORY=''")

    # Session info if available
    if data.session_id:
        print(f"SESSION_ID='{data.session_id}'")
    if data.goal:
        goal = data.goal.replace("'", "'\\''")
        print(f"SESSION_GOAL='{goal}'")


if __name__ == "__main__":
    main()
