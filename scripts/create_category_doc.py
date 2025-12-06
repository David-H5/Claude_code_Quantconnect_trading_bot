#!/usr/bin/env python3
"""Generate category research documents for multi-category upgrades.

Usage:
    python scripts/create_category_doc.py UPGRADE-014 3 "Fault Tolerance"
    python scripts/create_category_doc.py UPGRADE-014 3 "Fault Tolerance" --priority P0 --status "In Progress"
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


# Template path
TEMPLATE_PATH = Path("docs/research/templates/category_research_template.md")
OUTPUT_DIR = Path("docs/research")


def sanitize_name(name: str) -> str:
    """Convert name to uppercase with hyphens."""
    # Replace spaces and underscores with hyphens
    sanitized = re.sub(r"[\s_]+", "-", name.upper())
    # Remove any non-alphanumeric characters except hyphens
    sanitized = re.sub(r"[^A-Z0-9-]", "", sanitized)
    return sanitized


def generate_category_doc(
    upgrade_num: str,
    cat_num: int,
    category_name: str,
    priority: str = "P1",
    status: str = "Pending",
) -> Path:
    """Generate a category research document from template.

    Args:
        upgrade_num: Upgrade identifier (e.g., "014" or "UPGRADE-014")
        cat_num: Category number (1-12)
        category_name: Human-readable category name
        priority: Priority level (P0-P3)
        status: Implementation status

    Returns:
        Path to created document
    """
    # Normalize upgrade number
    if upgrade_num.startswith("UPGRADE-"):
        upgrade_num = upgrade_num.replace("UPGRADE-", "")

    # Sanitize category name for filename
    cat_name_sanitized = sanitize_name(category_name)

    # Generate filename
    filename = f"UPGRADE-{upgrade_num}-CAT{cat_num}-{cat_name_sanitized}-RESEARCH.md"
    output_path = OUTPUT_DIR / filename

    # Check if already exists
    if output_path.exists():
        print(f"WARNING: {output_path} already exists. Skipping.")
        return output_path

    # Load template
    if not TEMPLATE_PATH.exists():
        print(f"ERROR: Template not found at {TEMPLATE_PATH}")
        sys.exit(1)

    template = TEMPLATE_PATH.read_text()

    # Get current date
    today = datetime.now().strftime("%Y-%m-%d")

    # Replace placeholders
    content = template.replace("{UPGRADE_NUM}", upgrade_num)
    content = content.replace("{CAT_NUM}", str(cat_num))
    content = content.replace("{CATEGORY_NAME}", cat_name_sanitized)
    content = content.replace("{CATEGORY_TITLE}", category_name)
    content = content.replace("{PRIORITY}", priority.replace("P", ""))
    content = content.replace("{STATUS}", status)
    content = content.replace("{DATE}", today)
    content = content.replace("{MAIN_DOC}", f"UPGRADE-{upgrade_num}-AUTONOMOUS-AGENT-ENHANCEMENTS.md")

    # Write output
    output_path.write_text(content)
    print(f"Created: {output_path}")

    return output_path


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate category research documents for multi-category upgrades")
    parser.add_argument(
        "upgrade",
        help="Upgrade identifier (e.g., UPGRADE-014 or 014)",
    )
    parser.add_argument(
        "category_num",
        type=int,
        help="Category number (1-12)",
    )
    parser.add_argument(
        "category_name",
        help="Category name (e.g., 'Fault Tolerance')",
    )
    parser.add_argument(
        "--priority",
        default="P1",
        choices=["P0", "P1", "P2", "P3"],
        help="Priority level (default: P1)",
    )
    parser.add_argument(
        "--status",
        default="Pending",
        help="Implementation status (default: Pending)",
    )

    args = parser.parse_args()

    generate_category_doc(
        upgrade_num=args.upgrade,
        cat_num=args.category_num,
        category_name=args.category_name,
        priority=args.priority,
        status=args.status,
    )


if __name__ == "__main__":
    main()
