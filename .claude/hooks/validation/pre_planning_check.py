#!/usr/bin/env python3
"""Pre-Planning Conflict Check Hook.

Automatically runs before any upgrade/planning task to ensure
the agent has checked for existing implementations and duplications.

This hook enforces the consolidation-first policy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_planning_conflicts(proposed_feature: str) -> dict:
    """Run conflict analysis for a proposed feature."""
    try:
        from utils.codebase_analyzer import CodebaseAnalyzer

        analyzer = CodebaseAnalyzer(project_root=PROJECT_ROOT)
        return analyzer.check_planning_conflicts(proposed_feature)
    except ImportError:
        return {"error": "Codebase analyzer not available"}


def format_conflict_warning(result: dict) -> str:
    """Format conflict analysis as a warning message."""
    lines = [
        "",
        "=" * 60,
        "⚠️  PRE-PLANNING CONFLICT ANALYSIS",
        "=" * 60,
        "",
    ]

    risk = result.get("conflict_risk", "unknown")
    count = result.get("existing_count", 0)
    categories = result.get("matched_categories", [])

    if risk == "high":
        lines.append("🔴 HIGH CONFLICT RISK - Review existing code before proceeding!")
    elif risk == "medium":
        lines.append("🟡 MEDIUM CONFLICT RISK - Check for consolidation opportunities")
    else:
        lines.append("🟢 LOW CONFLICT RISK - Proceed with caution")

    lines.append("")
    lines.append(f"Found {count} existing implementations in categories: {', '.join(categories) or 'none'}")

    if result.get("existing_implementations"):
        lines.append("")
        lines.append("Existing implementations to review:")
        for impl in result.get("existing_implementations", [])[:10]:
            lines.append(f"  - {impl['file']}:{impl['line']} - {impl['name']}")

    if result.get("warnings"):
        lines.append("")
        lines.append("Warnings:")
        for warning in result.get("warnings", []):
            lines.append(f"  ⚠️  {warning}")

    lines.extend(
        [
            "",
            "REQUIRED ACTIONS:",
            "  1. Review each existing implementation above",
            "  2. Decide: EXTEND existing code OR CONSOLIDATE duplicates",
            "  3. Include deletion/merge plan in your upgrade guide",
            "  4. DO NOT create parallel implementations",
            "",
            "=" * 60,
            "",
        ]
    )

    return "\n".join(lines)


def get_consolidation_template() -> str:
    """Return the required template for upgrade guides."""
    return """
================================================================================
                    USE THE UPGRADE TEMPLATE
================================================================================

REQUIRED: Copy and use the upgrade template:

  cp .claude/templates/upgrade_template.md docs/upgrades/UPGRADE-XXX-name.md

The template includes all MANDATORY sections:

## 0. Prerequisites: Codebase Consolidation (BLOCKING)

   - 0.1 Conflict Analysis Results
   - 0.2 Existing Code Audit Table (EXTEND/DELETE/MERGE)
   - 0.3 Consolidation Plan (deletions, merges, extensions)
   - 0.4 Canonical Location (justified)
   - 0.5 Pre-Implementation Verification

## Phase Gates (Cannot proceed without passing)

   - Phase 0 Gate: Prerequisites complete
   - Phase 1 Gate: Consolidation complete (DELETE old code FIRST)

## Quick Reference - Existing Code Audit Table Format:

| File | Class/Function | Action | Justification |
|------|----------------|--------|---------------|
| path/existing.py | ExistingClass | EXTEND | Adding method fits here |
| path/old.py | OldFunction | DELETE | Replaced by new solution |
| a.py + b.py | Duplicates | MERGE | Consolidating |

ANTI-PATTERNS THAT WILL BE REJECTED:
  - Upgrade guide without Section 0
  - Plans creating parallel implementations (e.g., thing_v2.py)
  - Proposals without deletion timeline
  - "We'll consolidate later"

See: .claude/templates/upgrade_template.md for full template
See: docs/CONSOLIDATION_FIRST_POLICY.md for policy details
================================================================================
"""


PLANNING_KEYWORDS = [
    "upgrade",
    "plan",
    "implement",
    "add",
    "create",
    "build",
    "new feature",
    "new system",
    "new module",
    "refactor",
]


def should_run_check(user_message: str) -> bool:
    """Determine if this message triggers planning conflict check."""
    message_lower = user_message.lower()
    return any(keyword in message_lower for keyword in PLANNING_KEYWORDS)


def main():
    """Hook entry point."""
    # Read input from stdin (hook protocol)
    try:
        input_data = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        return 0

    user_message = input_data.get("message", "")

    if not should_run_check(user_message):
        return 0

    # Run conflict analysis
    result = check_planning_conflicts(user_message)

    if result.get("error"):
        print(f"Warning: {result['error']}", file=sys.stderr)
        return 0

    # Output warning if conflicts found
    if result.get("existing_count", 0) > 0:
        warning = format_conflict_warning(result)
        print(warning, file=sys.stderr)

        if result.get("conflict_risk") == "high":
            print(get_consolidation_template(), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
