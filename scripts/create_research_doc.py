#!/usr/bin/env python3
"""
Research Document Generator

Creates new research documents with proper naming conventions,
frontmatter, and cross-references.

Usage:
    python scripts/create_research_doc.py "Evaluation Framework" --topic evaluation
    python scripts/create_research_doc.py "LLM Sentiment" --topic llm --upgrade UPGRADE-014

Options:
    --topic     Topic category (required)
    --upgrade   Related upgrade number(s)
    --tags      Additional tags
    --type      Document type: research, summary, guide (default: research)
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


# Valid topic categories
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

# Document templates
TEMPLATES = {
    "research": """---
title: "{title}"
topic: {topic}
related_upgrades: [{upgrades}]
related_docs: []
tags: [{tags}]
created: {date}
updated: {date}
---

# {title} Research

## 📋 Research Overview

**Date**: {date}
**Scope**: [What was researched]
**Focus**: [Specific areas]
**Result**: [Summary of deliverables]

---

## 🎯 Research Objectives

1. [Objective 1]
2. [Objective 2]
3. [Objective 3]

---

## 📊 Research Phases

### Phase 1: [Name]

**Search Date**: {date} at [TIME] EST
**Search Queries**:
- "[query 1]"
- "[query 2]"

**Key Sources**:

1. [Source Title (Published: Month Year)](URL)
2. [Source Title (Published: Month Year)](URL)

**Key Discoveries**:

- [Finding 1]
- [Finding 2]

**Applied**: [What was implemented]

---

### Phase 2: [Name]

**Search Date**: {date} at [TIME] EST
**Search Queries**: [queries]

**Key Sources**:

1. [Source (Published: Date)](URL)

**Key Discoveries**: [Findings]
**Applied**: [Implementation]

---

## 🔑 Critical Discoveries

[Most important findings with impact assessment]

---

## 💾 Research Deliverables

| Document | Size | Purpose |
|----------|------|---------|
| [File] | [Size] | [Purpose] |

---

## 📝 Change Log

| Date | Change |
|------|--------|
| {date} | Initial research document created |
""",
    "summary": """---
title: "{title}"
topic: {topic}
related_upgrades: [{upgrades}]
related_docs: []
tags: [{tags}]
created: {date}
updated: {date}
---

# {title} Summary

## Executive Summary

[Brief overview of the topic and key findings]

---

## Key Points

1. **[Point 1]**: [Description]
2. **[Point 2]**: [Description]
3. **[Point 3]**: [Description]

---

## Recommendations

- [Recommendation 1]
- [Recommendation 2]

---

## Related Documents

- [Related Doc 1](related_doc_1.md)
- [Related Doc 2](related_doc_2.md)

---

## Change Log

| Date | Change |
|------|--------|
| {date} | Initial summary created |
""",
    "guide": """---
title: "{title}"
topic: {topic}
related_upgrades: [{upgrades}]
related_docs: []
tags: [{tags}]
created: {date}
updated: {date}
---

# {title} Guide

## Overview

[Purpose of this guide]

---

## Prerequisites

- [Prerequisite 1]
- [Prerequisite 2]

---

## Implementation Steps

### Step 1: [Name]

[Description]

```python
# Example code
```

### Step 2: [Name]

[Description]

---

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| [Setting] | [Value] | [Description] |

---

## Validation

- [ ] [Check 1]
- [ ] [Check 2]

---

## Troubleshooting

### Issue: [Common Issue]

**Cause**: [Why it happens]
**Solution**: [How to fix]

---

## Related Documents

- [Related Doc 1](related_doc_1.md)

---

## Change Log

| Date | Change |
|------|--------|
| {date} | Initial guide created |
""",
}


def normalize_topic_name(name: str) -> str:
    """Convert a topic name to SCREAMING_SNAKE_CASE filename."""
    # Remove special characters
    name = re.sub(r"[^\w\s-]", "", name)
    # Convert to uppercase and replace spaces with underscores
    name = name.upper().replace(" ", "_").replace("-", "_")
    # Remove multiple underscores
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def create_research_doc(
    title: str,
    topic: str,
    doc_type: str = "research",
    upgrades: list[str] | None = None,
    tags: list[str] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Create a new research document."""
    # Validate topic
    if topic.lower() not in VALID_TOPICS:
        print(f"Warning: '{topic}' is not a standard topic. Valid topics: {', '.join(VALID_TOPICS)}")

    # Generate filename
    normalized_name = normalize_topic_name(title)

    if doc_type == "research":
        filename = f"{normalized_name}_RESEARCH.md"
    elif doc_type == "summary":
        filename = f"{normalized_name}_SUMMARY.md"
    elif doc_type == "guide":
        filename = f"{normalized_name}_UPGRADE_GUIDE.md"
    else:
        raise ValueError(f"Unknown document type: {doc_type}")

    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        output_dir = project_root / "docs" / "research"

    output_path = output_dir / filename

    # Check if file exists
    if output_path.exists():
        print(f"Error: File already exists: {output_path}")
        return output_path

    # Prepare template variables
    date_str = datetime.now().strftime("%Y-%m-%d")
    upgrade_str = ", ".join(upgrades) if upgrades else ""
    tag_list = tags or [topic.lower()]
    tag_str = ", ".join(tag_list)

    # Get template
    template = TEMPLATES.get(doc_type, TEMPLATES["research"])

    # Fill template
    content = template.format(
        title=title,
        topic=topic.lower(),
        upgrades=upgrade_str,
        tags=tag_str,
        date=date_str,
    )

    # Write file
    output_path.write_text(content, encoding="utf-8")
    print(f"✅ Created: {output_path}")

    # Remind about index updates
    print("\n📝 Next steps:")
    print("   1. Fill in the research content")
    print("   2. Add to docs/research/README.md quick reference")
    if upgrades:
        print("   3. Update docs/research/UPGRADE_INDEX.md")
    print("   4. Run: python scripts/validate_research_docs.py")

    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create new research document with proper naming")
    parser.add_argument(
        "title",
        type=str,
        help="Document title (e.g., 'Evaluation Framework')",
    )
    parser.add_argument(
        "--topic",
        "-t",
        type=str,
        required=True,
        help=f"Topic category: {', '.join(VALID_TOPICS)}",
    )
    parser.add_argument(
        "--upgrade",
        "-u",
        type=str,
        nargs="+",
        help="Related upgrade number(s) (e.g., UPGRADE-014)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="Additional tags",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="research",
        choices=["research", "summary", "guide"],
        help="Document type (default: research)",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Output directory (default: docs/research)",
    )

    args = parser.parse_args()

    try:
        create_research_doc(
            title=args.title,
            topic=args.topic,
            doc_type=args.type,
            upgrades=args.upgrade,
            tags=args.tags,
            output_dir=args.dir,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
