#!/usr/bin/env python3
"""Hook to validate category documentation exists before allowing progress updates.

This hook is triggered when claude-progress.txt is modified and validates that:
1. Any category marked complete has a corresponding research document
2. Multi-category upgrades have all required documents

Usage:
    Called automatically by Claude hooks when progress file changes.
    Can also be run manually: python .claude/hooks/validate_category_docs.py
"""

import json
import os
import re
import sys
from pathlib import Path


# Configuration
PROGRESS_FILE = Path("claude-progress.txt")
RESEARCH_DIR = Path("docs/research")

# Category mapping for UPGRADE-014
UPGRADE_014_CATEGORIES = {
    1: ("Architecture", "ARCHITECTURE"),
    2: ("Observability", "OBSERVABILITY"),
    3: ("Fault Tolerance", "FAULT-TOLERANCE"),
    4: ("Memory Management", "MEMORY-MANAGEMENT"),
    5: ("Safety Guardrails", "SAFETY-GUARDRAILS"),
    6: ("Cost Optimization", "COST-OPTIMIZATION"),
    7: ("State Persistence", "STATE-PERSISTENCE"),
    8: ("Testing & Simulation", "TESTING-SIMULATION"),
    9: ("Self-Improvement", "SELF-IMPROVEMENT"),
    10: ("Workspace Management", "WORKSPACE-MANAGEMENT"),
    11: ("Overnight Sessions", "OVERNIGHT-SESSIONS"),
    12: ("Claude Code Specific", "CLAUDE-CODE-SPECIFIC"),
}


def get_expected_doc_path(upgrade_num: str, cat_num: int, cat_name: str) -> Path:
    """Get the expected path for a category research document."""
    return RESEARCH_DIR / f"UPGRADE-{upgrade_num}-CAT{cat_num}-{cat_name}-RESEARCH.md"


def parse_progress_file() -> dict:
    """Parse claude-progress.txt to find completed categories."""
    if not PROGRESS_FILE.exists():
        return {"upgrade": None, "completed_categories": [], "in_progress_categories": []}

    content = PROGRESS_FILE.read_text()

    # Find upgrade number
    upgrade_match = re.search(r"UPGRADE-(\d+)", content)
    upgrade_num = upgrade_match.group(1) if upgrade_match else None

    # Find completed categories (marked with [x])
    completed = []
    in_progress = []

    # Pattern: ## CATEGORY N: Name ... - COMPLETED or - IN PROGRESS
    category_pattern = re.compile(
        r"##\s*CATEGORY\s*(\d+)[:\s]+([^(\n]+?)(?:\s*\([^)]+\))?\s*-\s*(COMPLETED|IN PROGRESS|PENDING)",
        re.IGNORECASE,
    )

    for match in category_pattern.finditer(content):
        cat_num = int(match.group(1))
        status = match.group(3).upper()

        if status == "COMPLETED":
            completed.append(cat_num)
        elif status == "IN PROGRESS":
            in_progress.append(cat_num)

    return {
        "upgrade": upgrade_num,
        "completed_categories": completed,
        "in_progress_categories": in_progress,
    }


def validate_category_docs(upgrade_num: str, categories: list) -> dict:
    """Validate that category research documents exist.

    Args:
        upgrade_num: The upgrade number (e.g., "014")
        categories: List of category numbers to validate

    Returns:
        Dict with validation results
    """
    missing = []
    found = []

    for cat_num in categories:
        if cat_num not in UPGRADE_014_CATEGORIES:
            continue

        _, cat_name = UPGRADE_014_CATEGORIES[cat_num]
        expected_path = get_expected_doc_path(upgrade_num, cat_num, cat_name)

        if expected_path.exists():
            found.append((cat_num, str(expected_path)))
        else:
            missing.append((cat_num, str(expected_path)))

    return {
        "valid": len(missing) == 0,
        "found": found,
        "missing": missing,
    }


def main() -> int:
    """Main entry point.

    Returns:
        0 if validation passes, 1 if missing documents, 2 on error
    """
    # Check for tool input from Claude hook system (reserved for future use)
    _ = os.environ.get("CLAUDE_TOOL_INPUT", "")

    # Parse progress file
    progress = parse_progress_file()

    if not progress["upgrade"]:
        print("INFO: No upgrade number found in progress file.")
        return 0

    upgrade_num = progress["upgrade"]

    # Validate completed categories have docs
    completed_validation = validate_category_docs(upgrade_num, progress["completed_categories"])

    # Check in-progress categories (warn but don't fail)
    in_progress_validation = validate_category_docs(upgrade_num, progress["in_progress_categories"])

    # Report results
    result = {
        "upgrade": f"UPGRADE-{upgrade_num}",
        "completed_categories": {
            "count": len(progress["completed_categories"]),
            "valid": completed_validation["valid"],
            "missing_docs": completed_validation["missing"],
        },
        "in_progress_categories": {
            "count": len(progress["in_progress_categories"]),
            "missing_docs": in_progress_validation["missing"],
        },
    }

    # Output for Claude
    if completed_validation["missing"]:
        print("ERROR: Missing documentation for completed categories:")
        for cat_num, path in completed_validation["missing"]:
            cat_title, _ = UPGRADE_014_CATEGORIES[cat_num]
            print(f"  - Category {cat_num} ({cat_title}): {path}")
        print("\nACTION REQUIRED: Create these documents before marking categories complete.")
        print("Use: python scripts/create_category_doc.py UPGRADE-014 <cat_num> '<name>'")

        # Output JSON for machine parsing
        print(f"\n[VALIDATION_RESULT]: {json.dumps(result)}")
        return 1

    if in_progress_validation["missing"]:
        print("WARNING: Missing documentation for in-progress categories:")
        for cat_num, path in in_progress_validation["missing"]:
            cat_title, _ = UPGRADE_014_CATEGORIES[cat_num]
            print(f"  - Category {cat_num} ({cat_title}): {path}")
        print("\nREMINDER: Create documentation as you implement.")

    if completed_validation["found"]:
        print(f"OK: All {len(completed_validation['found'])} completed categories have documentation.")

    # Output JSON for machine parsing
    print(f"\n[VALIDATION_RESULT]: {json.dumps(result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
