#!/usr/bin/env python3
"""
Research Document Validation Hook

This hook validates research documents after they are created or modified.
Ensures proper naming conventions, frontmatter, and cross-references.

Triggers on: Write/Edit to docs/research/ directory
Purpose: Enforce research documentation standards automatically
"""

import json
import re
import sys
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Valid naming patterns for research documents
VALID_PATTERNS = [
    r"^[A-Z][A-Z0-9_]+_RESEARCH\.md$",
    r"^[A-Z][A-Z0-9_]+_SUMMARY\.md$",
    r"^[A-Z][A-Z0-9_]+_UPGRADE_GUIDE\.md$",
    r"^UPGRADE-\d{3}-[A-Z][A-Z0-9-]+\.md$",
    r"^README\.md$",
    r"^NAMING_CONVENTION\.md$",
    r"^UPGRADE_INDEX\.md$",
]

# Required frontmatter fields
REQUIRED_FRONTMATTER = ["title", "topic", "tags", "created"]

# Valid topics
VALID_TOPICS = [
    "quantconnect",
    "evaluation",
    "llm",
    "workflow",
    "prompts",
    "autonomous",
    "agents",
    "sentiment",
    "integration",
    "general",
]


def parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from markdown content."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}

    result = {}
    yaml_str = match.group(1)
    current_key = None
    current_list = []

    for line in yaml_str.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line and not line.startswith("-"):
            if current_key and current_list:
                result[current_key] = current_list
                current_list = []

            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            if value.startswith("[") and value.endswith("]"):
                items = value[1:-1].split(",")
                result[key] = [item.strip().strip("'\"") for item in items if item.strip()]
            elif value:
                result[key] = value.strip("'\"")
            else:
                current_key = key
        elif line.startswith("-") and current_key:
            item = line[1:].strip().strip("'\"")
            current_list.append(item)

    if current_key and current_list:
        result[current_key] = current_list

    return result


def validate_research_doc(file_path: str, content: str) -> dict:
    """Validate a research document and return issues found."""
    issues = []
    warnings = []
    suggestions = []

    filename = Path(file_path).name

    # Check if this is a research document
    if not file_path.startswith("docs/research/") and "/docs/research/" not in file_path:
        return {"valid": True, "issues": [], "warnings": [], "suggestions": []}

    # Skip non-markdown files
    if not filename.endswith(".md"):
        return {"valid": True, "issues": [], "warnings": [], "suggestions": []}

    # Check naming convention
    is_valid_name = any(re.match(pattern, filename) for pattern in VALID_PATTERNS)
    if not is_valid_name:
        # Check for common violations
        if filename.startswith("UPGRADE_") and "RESEARCH" in filename:
            issues.append(f"Invalid naming: '{filename}' uses upgrade number in name")
            topic = filename.replace("UPGRADE_", "").replace("_RESEARCH.md", "")
            topic = re.sub(r"^\d+_?", "", topic)
            suggestions.append(f"Rename to: {topic}_RESEARCH.md (add upgrade to frontmatter)")
        else:
            issues.append(f"Invalid naming: '{filename}' doesn't match conventions")
            suggestions.append("Use format: [TOPIC]_RESEARCH.md (SCREAMING_SNAKE_CASE)")

    # Check frontmatter
    frontmatter = parse_frontmatter(content)
    if not frontmatter:
        if filename.endswith("_RESEARCH.md"):
            issues.append("Missing YAML frontmatter")
            suggestions.append("Add frontmatter with: title, topic, tags, created fields")
    else:
        # Check required fields
        missing = [f for f in REQUIRED_FRONTMATTER if f not in frontmatter]
        if missing:
            warnings.append(f"Missing frontmatter fields: {', '.join(missing)}")

        # Validate topic
        topic = frontmatter.get("topic", "").lower()
        if topic and topic not in VALID_TOPICS:
            warnings.append(f"Invalid topic '{topic}'. Valid: {', '.join(VALID_TOPICS)}")

    # Check for timestamp requirements in research docs
    if filename.endswith("_RESEARCH.md"):
        if "Search Date" not in content and "**Search Date**" not in content:
            warnings.append("No search timestamps found")
            suggestions.append("Add search date/time for each research phase")

        if "(Published:" not in content and "(Updated:" not in content:
            warnings.append("No source publication dates found")
            suggestions.append("Add publication dates to sources: [Title (Published: Month Year)](URL)")

    # Check minimum content for research docs
    word_count = len(content.split())
    if filename.endswith("_RESEARCH.md") and word_count < 500:
        warnings.append(f"Document short ({word_count} words, minimum 500 recommended)")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "suggestions": suggestions,
    }


def format_output(validation_result: dict, filename: str) -> str:
    """Format validation output for display."""
    lines = []

    if not validation_result["valid"]:
        lines.append(f"Research Doc Validation Failed: {filename}")
        for issue in validation_result["issues"]:
            lines.append(f"  ERROR: {issue}")

    if validation_result["warnings"]:
        if not lines:
            lines.append(f"Research Doc Warnings: {filename}")
        for warning in validation_result["warnings"]:
            lines.append(f"  WARNING: {warning}")

    if validation_result["suggestions"]:
        for suggestion in validation_result["suggestions"]:
            lines.append(f"  TIP: {suggestion}")

    if not lines:
        return ""

    lines.append("")
    lines.append("Run: python scripts/validate_research_docs.py --fix for auto-fixes")
    lines.append("See: docs/research/NAMING_CONVENTION.md for rules")

    return "\n".join(lines)


def main():
    """Main hook entry point."""
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Only check for Write/Edit to research docs
    if tool_name not in ["Write", "Edit"]:
        sys.exit(0)

    file_path = tool_input.get("file_path", "")

    # Check if it's a research document
    if "docs/research/" not in file_path:
        sys.exit(0)

    # Get content (for Write) or read file (for Edit)
    content = tool_input.get("content", "")
    if not content and tool_name == "Edit":
        try:
            full_path = PROJECT_ROOT / file_path.lstrip("/")
            if full_path.exists():
                content = full_path.read_text(encoding="utf-8")
        except Exception:
            sys.exit(0)

    if not content:
        sys.exit(0)

    # Validate
    result = validate_research_doc(file_path, content)

    # Output warnings/errors
    output = format_output(result, Path(file_path).name)
    if output:
        print(output)

    # Allow operation to proceed (hook is advisory, not blocking)
    sys.exit(0)


if __name__ == "__main__":
    main()
