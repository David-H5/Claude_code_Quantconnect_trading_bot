#!/usr/bin/env python3
"""
Research Saver - Intelligent Documentation Tool

Automatically saves research, insights, ideas, and upgrade guides to the
appropriate locations with proper naming conventions and conflict checking.

Features:
- Auto-detect document type (research, upgrade, insight, guide)
- Check for naming conflicts
- Follow project naming conventions
- Place files in correct directories
- Update indexes automatically
- Generate from templates
- Track document lineage

Usage:
    # CLI
    python research_saver.py create --type upgrade --name "Multi-Agent" --number 17
    python research_saver.py create --type research --topic "Agent Memory"
    python research_saver.py check-name "UPGRADE-017-SOMETHING.md"
    python research_saver.py list-available-numbers
    python research_saver.py template upgrade

    # From Claude Code
    /save-research upgrade "Multi-Agent Orchestration" --number 17
    /save-research research "Agent Memory Patterns"
    /save-research insight "Context window optimization"
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESEARCH_DIR = PROJECT_ROOT / "docs" / "research"
TEMPLATES_DIR = PROJECT_ROOT / ".claude" / "templates"
RESEARCH_INDEX = RESEARCH_DIR / "README.md"

# Document type configurations
DOC_CONFIGS = {
    "upgrade": {
        "directory": RESEARCH_DIR,
        "pattern": r"UPGRADE-(\d+)(?:-[A-Z0-9-]+)?\.md",
        "prefix": "UPGRADE-",
        "example": "UPGRADE-017-MULTI-AGENT.md",
        "required_sections": [
            "Overview",
            "Objectives",
            "Implementation Checklist",
            "Progress Tracking",
            "References",
            "Change Log",
        ],
    },
    "research": {
        "directory": RESEARCH_DIR,
        "pattern": r"([A-Z][A-Z0-9_-]+)-RESEARCH\.md",
        "suffix": "-RESEARCH.md",
        "example": "AGENT-MEMORY-RESEARCH.md",
        "required_sections": [
            "Research Overview",
            "Research Objectives",
            "Key Discoveries",
            "Research Deliverables",
            "Change Log",
        ],
    },
    "category": {
        "directory": RESEARCH_DIR,
        "pattern": r"UPGRADE-(\d+)-CAT(\d+)-([A-Z-]+)-RESEARCH\.md",
        "example": "UPGRADE-014-CAT3-FAULT-TOLERANCE-RESEARCH.md",
        "required_sections": ["Category Overview", "Research Phases", "Implementation Tasks", "Progress", "References"],
    },
    "insight": {
        "directory": PROJECT_ROOT / "docs" / "insights",
        "pattern": r"INSIGHT-(\d{4}-\d{2}-\d{2})-([A-Z-]+)\.md",
        "example": "INSIGHT-2025-12-04-AGENT-HANDOFFS.md",
        "required_sections": ["Insight Summary", "Context", "Key Finding", "Application", "Related Work"],
    },
    "guide": {
        "directory": PROJECT_ROOT / "docs" / "guides",
        "pattern": r"([A-Z][A-Z0-9_-]+)-GUIDE\.md",
        "suffix": "-GUIDE.md",
        "example": "MULTI-AGENT-GUIDE.md",
        "required_sections": [
            "Overview",
            "Prerequisites",
            "Quick Start",
            "Detailed Usage",
            "Examples",
            "Troubleshooting",
            "References",
        ],
    },
}


class DocType(Enum):
    UPGRADE = "upgrade"
    RESEARCH = "research"
    CATEGORY = "category"
    INSIGHT = "insight"
    GUIDE = "guide"


@dataclass
class DocumentMetadata:
    doc_type: DocType
    name: str
    filename: str
    filepath: Path
    created_at: datetime
    version: str = "1.0.0"
    status: str = "draft"
    priority: str = "P1"
    estimated_effort: str = "TBD"
    tags: list[str] = field(default_factory=list)
    related_docs: list[str] = field(default_factory=list)
    upgrade_number: int | None = None
    category_number: int | None = None


# =============================================================================
# Naming Convention Checker
# =============================================================================


class NamingChecker:
    """Check and validate document names against conventions."""

    def __init__(self):
        self.existing_docs = self._scan_existing_docs()

    def _scan_existing_docs(self) -> dict[str, list[str]]:
        """Scan all existing documentation files."""
        docs = {dtype: [] for dtype in DOC_CONFIGS}

        for doc_type, config in DOC_CONFIGS.items():
            directory = config["directory"]
            if directory.exists():
                pattern = config["pattern"]
                for file in directory.glob("*.md"):
                    if re.match(pattern, file.name):
                        docs[doc_type].append(file.name)

        return docs

    def get_next_upgrade_number(self) -> int:
        """Get the next available upgrade number."""
        existing_numbers = []
        pattern = DOC_CONFIGS["upgrade"]["pattern"]

        for filename in self.existing_docs["upgrade"]:
            match = re.match(pattern, filename)
            if match:
                existing_numbers.append(int(match.group(1)))

        if not existing_numbers:
            return 1
        return max(existing_numbers) + 1

    def check_conflict(self, filename: str, doc_type: str) -> tuple[bool, str | None]:
        """Check if filename conflicts with existing docs."""
        if filename in self.existing_docs.get(doc_type, []):
            return True, f"File '{filename}' already exists"

        # Check for similar names (fuzzy matching)
        base_name = re.sub(r"[-_\d]+", "", filename.upper())
        for existing in self.existing_docs.get(doc_type, []):
            existing_base = re.sub(r"[-_\d]+", "", existing.upper())
            if base_name == existing_base and filename != existing:
                return True, f"Similar file exists: '{existing}'"

        return False, None

    def validate_name(self, filename: str, doc_type: str) -> tuple[bool, list[str]]:
        """Validate filename against naming conventions."""
        errors = []
        config = DOC_CONFIGS.get(doc_type)

        if not config:
            errors.append(f"Unknown document type: {doc_type}")
            return False, errors

        pattern = config["pattern"]
        if not re.match(pattern, filename):
            errors.append(f"Filename doesn't match pattern: {pattern}")
            errors.append(f"Example: {config['example']}")

        # Check character conventions
        if not filename.isupper() or " " in filename:
            # Allow mixed case only in specific patterns
            if doc_type not in ["insight"]:
                if not re.match(r"^[A-Z0-9_-]+\.md$", filename):
                    errors.append("Use UPPERCASE with hyphens/underscores, no spaces")

        return len(errors) == 0, errors

    def suggest_name(self, title: str, doc_type: str, number: int | None = None) -> str:
        """Suggest a proper filename based on title and type."""
        # Normalize title
        normalized = re.sub(r"[^a-zA-Z0-9\s]", "", title)
        normalized = normalized.upper().replace(" ", "-")
        normalized = re.sub(r"-+", "-", normalized).strip("-")

        if doc_type == "upgrade":
            if number is None:
                number = self.get_next_upgrade_number()
            return f"UPGRADE-{number:03d}-{normalized}.md"

        elif doc_type == "research":
            return f"{normalized}-RESEARCH.md"

        elif doc_type == "insight":
            date_str = datetime.now().strftime("%Y-%m-%d")
            return f"INSIGHT-{date_str}-{normalized}.md"

        elif doc_type == "guide":
            return f"{normalized}-GUIDE.md"

        elif doc_type == "category":
            # Requires both upgrade and category numbers
            return f"UPGRADE-{number:03d}-CAT{1}-{normalized}-RESEARCH.md"

        return f"{normalized}.md"


# =============================================================================
# Template Generator
# =============================================================================


class TemplateGenerator:
    """Generate document templates."""

    @staticmethod
    def get_upgrade_template(
        title: str, number: int, description: str = "", priority: str = "P1", effort: str = ""
    ) -> str:
        """Generate streamlined upgrade template for AI agents.

        Enforces mandatory timestamping and clear task tracking.
        """
        date = datetime.now().strftime("%Y-%m-%d")
        time_now = datetime.now().strftime("%H:%M")
        tz = "EST"
        return f"""# UPGRADE-{number:03d}: {title}

## 📋 Overview

**Created**: {date} at {time_now} {tz}
**Agent**: Claude Code
**Status**: Planning | In Progress | Complete | Blocked
**Priority**: {priority}

### Summary

{description or "[2-3 sentences explaining what this upgrade accomplishes and why]"}

---

## 🎯 Scope

### Goals

| # | Goal | Success Metric | Status |
|---|------|----------------|--------|
| 1 | [Goal] | [Measurable outcome] | ⬜ |
| 2 | [Goal] | [Measurable outcome] | ⬜ |

### Non-Goals

- [What this upgrade will NOT do]
- [Explicitly out of scope]

---

## 📊 Research

### Phase 1: [Topic]

**Search Timestamp**: {date} at {time_now} {tz}  ← **REQUIRED**

**Queries**: "[query1]", "[query2]"

**Sources**:

| # | Source | Published | Key Finding |
|---|--------|-----------|-------------|
| 1 | [Title](URL) | YYYY-MM | [Finding] |

> **TIMESTAMPING RULES**:
> - Search timestamp: Exact time when search was performed
> - Published date: Use `YYYY-MM`, `~YYYY` for estimates, `Unknown` if undetermined

**Applied**: [What was implemented from this research]

---

## ✅ Implementation Checklist

### Status Legend

| Symbol | Meaning |
|--------|---------|
| ⬜ | Not started |
| 🔄 | In progress |
| ✅ | Complete |
| ⏸️ | Blocked |

---

### Phase 1: [Name]

**Goal**: [What this phase accomplishes]

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 1.1 | [Task description] | `path/file.py` | ⬜ |
| 1.2 | [Task description] | `path/file.py` | ⬜ |
| 1.3 | [Task description] | `path/file.py` | ⬜ |

---

### Phase 2: [Name]

**Goal**: [What this phase accomplishes]
**Depends on**: Phase 1 complete

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 2.1 | [Task description] | `path/file.py` | ⬜ |
| 2.2 | [Task description] | `path/file.py` | ⬜ |

---

### Phase 3: Testing

**Goal**: Validate implementation

| # | Task | Target | Status |
|---|------|--------|--------|
| 3.1 | Unit tests | >80% coverage | ⬜ |
| 3.2 | Integration tests | All pass | ⬜ |
| 3.3 | Run linter | No errors | ⬜ |

---

### Phase 4: Documentation

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 4.1 | Update CLAUDE.md | `CLAUDE.md` | ⬜ |
| 4.2 | Add docstrings | New files | ⬜ |
| 4.3 | Remove TODOs/debug code | All files | ⬜ |

---

## 📁 Files

### New Files

| File | Purpose |
|------|---------|
| `path/to/file.py` | [Purpose] |

### Modified Files

| File | Changes |
|------|---------|
| `path/to/file.py` | [What changes] |

---

## 📊 Progress

| Phase | Tasks | Done | Status |
|-------|-------|------|--------|
| 1: [Name] | X | 0 | ⬜ |
| 2: [Name] | X | 0 | ⬜ |
| 3: Testing | X | 0 | ⬜ |
| 4: Docs | X | 0 | ⬜ |
| **Total** | **XX** | **0** | **0%** |

---

## ✔️ Definition of Done

### Per Task

- [ ] Code compiles without errors
- [ ] Unit tests added and passing
- [ ] No linting errors

### Per Phase

- [ ] All tasks in phase complete
- [ ] All tests passing

### Overall

- [ ] All phases complete
- [ ] Coverage ≥ 70%
- [ ] CLAUDE.md updated
- [ ] No TODO comments in code

---

## 🔙 Rollback

**Trigger**: If critical functionality breaks

```bash
git revert HEAD --no-edit
```

---

## 📝 Change Log

| Date | Change |
|------|--------|
| {date} | Initial creation |

---

## 📊 Tags

`upgrade-{number:03d}` `[category]`
"""

    @staticmethod
    def get_research_template(topic: str, scope: str = "") -> str:
        """Generate streamlined research document template for AI agents.

        Enforces mandatory timestamping for search times and source publication dates.
        """
        date = datetime.now().strftime("%Y-%m-%d")
        time_now = datetime.now().strftime("%H:%M")
        tz = "EST"
        normalized_topic = topic.lower().replace(" ", "-")
        return f"""# {topic} Research

## 📋 Overview

**Search Date**: {date} at {time_now} {tz}
**Agent**: Claude Code
**Scope**: {scope or f"Research into {topic}"}
**Result**: [Deliverables summary]

---

## 🎯 Research Questions

1. [Primary question]
2. [Secondary question]

---

## 📊 Research Phases

### Phase 1: [Name]

**Search Timestamp**: {date} at {time_now} {tz}  ← **REQUIRED**

**Queries**:
- "[Query 1]"
- "[Query 2]"

**Sources**:

| # | Source | Published | Key Finding |
|---|--------|-----------|-------------|
| 1 | [Title](URL) | YYYY-MM or "~YYYY" if estimated | [Finding] |
| 2 | [Title](URL) | YYYY-MM or "Unknown" | [Finding] |

> **TIMESTAMPING RULES**:
> - Search timestamp: Exact time when search was performed
> - Published date: Use `YYYY-MM` format, `~YYYY` for estimates, `Unknown` if cannot determine
> - Sources older than 2 years: Note "[DATED]" and verify still applicable

**Discoveries**:

1. **[Title]**: [Description]
   - Confidence: High/Medium/Low
   - Evidence: [Quote or data point]

**Applied**: [What was implemented]

---

## 🔑 Critical Discoveries

### Discovery 1: [Title]

**TL;DR**: [One sentence - lead with conclusion]

**Source**: [Reference](URL) (Published: YYYY-MM)
**Impact**: High/Medium/Low
**Confidence**: High (3+ sources) / Medium (2 sources) / Low (1 source)

**Application**: [How this applies to our project]

---

## ✅ Best Practices

| Practice | Benefit | Source |
|----------|---------|--------|
| [Practice] | [Why] | [Source](URL) (YYYY-MM) |

---

## 🚫 Anti-Patterns

| Anti-Pattern | Why Bad | Alternative |
|--------------|---------|-------------|
| [Pattern] | [Why] | [Better approach] |

---

## 💾 Deliverables

| Deliverable | Location | Status |
|-------------|----------|--------|
| [Item] | `path/to/file` | ✅/⬜ |

---

## 📝 Change Log

| Date | Change |
|------|--------|
| {date} | Initial research |

---

## 📊 Tags

`{normalized_topic}` `research`
"""

    @staticmethod
    def get_insight_template(title: str, context: str = "") -> str:
        """Generate streamlined insight document template for AI agents.

        Enforces mandatory timestamping for discovery time and source publication dates.
        """
        date = datetime.now().strftime("%Y-%m-%d")
        time_now = datetime.now().strftime("%H:%M")
        tz = "EST"
        normalized_title = title.lower().replace(" ", "-")
        return f"""# Insight: {title}

## 📋 Summary

**Date**: {date} at {time_now} {tz}
**Agent**: Claude Code
**Category**: Development | Architecture | Performance | Security | Trading
**Impact**: High | Medium | Low
**Actionable**: Yes | No

### One-Line Summary

[Single sentence describing the insight]

---

## 🔍 Context

**What prompted this**: {context or "[Problem being solved or research being conducted]"}

**Discovery timestamp**: {date} at {time_now} {tz}

---

## 💡 Key Finding

### The Insight

[2-3 sentences describing what was discovered]

### Why It Matters

[Why this insight is important]

### Evidence

| Evidence | Source | Published |
|----------|--------|-----------|
| [Finding/quote] | [Source](URL) | YYYY-MM |

> **TIMESTAMPING RULES**:
> - Discovery timestamp: When the insight was identified
> - Source published: Use `YYYY-MM`, `~YYYY` for estimates, `Unknown` if undetermined

### Confidence

| Factor | Rating |
|--------|--------|
| Source reliability | High/Medium/Low |
| Reproducibility | High/Medium/Low |
| **Overall** | **High/Medium/Low** |

---

## 🎯 Application

### How to Apply

1. [Step 1]
2. [Step 2]
3. [Step 3]

### Where to Apply

| Location | Priority |
|----------|----------|
| `path/to/file.py` | P0/P1/P2 |

### Code Example (if applicable)

```python
# Example implementation
```

---

## ⚠️ Caveats

- [When NOT to apply this insight]
- [Edge cases]

---

## 📝 Follow-up

| Action | Priority | Status |
|--------|----------|--------|
| [Action item] | P0/P1/P2 | ⬜ |

---

## 🔗 Related

- [Related doc](path) - [Description]

---

## 📊 Tags

`{normalized_title}` `insight`
"""

    @staticmethod
    def get_guide_template(title: str, description: str = "") -> str:
        """Generate streamlined guide document template for AI agents."""
        date = datetime.now().strftime("%Y-%m-%d")
        normalized_title = title.lower().replace(" ", "-")
        return f"""# {title} Guide

## 📋 Overview

**Last Updated**: {date}
**Agent**: Claude Code
**Difficulty**: Beginner | Intermediate | Advanced
**Time Required**: [ESTIMATED_TIME]

### What This Covers

{description or "[One paragraph description]"}

### Prerequisites

- [Prerequisite 1]
- [Prerequisite 2]

---

## 🚀 Quick Start

```bash
# 1. [First step]
[command]

# 2. [Second step]
[command]

# 3. Verify
[verification command]
```

---

## 📖 Detailed Steps

### Step 1: [Title]

**Purpose**: [What this accomplishes]

```bash
[command or code]
```

**Expected output**:

```text
[what you should see]
```

### Step 2: [Title]

**Purpose**: [What this accomplishes]

```python
# Code example
```

### Step 3: [Title]

[Explanation]

---

## 💡 Examples

### Example 1: [Basic Use Case]

**Scenario**: [When you'd use this]

```python
# Implementation
```

**Output**:

```text
[expected result]
```

### Example 2: [Advanced Use Case]

**Scenario**: [When you'd use this]

```python
# Implementation
```

---

## ⚠️ Troubleshooting

### [Common Issue 1]

**Symptom**: [What you see]

**Fix**:

```bash
[fix command]
```

### [Common Issue 2]

**Symptom**: [Error message]

**Fix**: [Solution]

---

## 🔧 Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `[option]` | `[value]` | [What it does] |

---

## 🔗 References

- [Official Docs](URL) - [Description]
- [Related Guide](path) - [Description]

---

## 📝 Change Log

| Date | Change |
|------|--------|
| {date} | Initial creation |

---

## 📊 Tags

`{normalized_title}` `guide`
"""


# =============================================================================
# Document Saver
# =============================================================================


class ResearchSaver:
    """Main class for saving research documents."""

    def __init__(self):
        self.checker = NamingChecker()
        self.generator = TemplateGenerator()

    def create_document(
        self, doc_type: str, title: str, content: str | None = None, number: int | None = None, **kwargs
    ) -> tuple[bool, str, Path]:
        """
        Create a new research document.

        Returns:
            Tuple of (success, message, filepath)
        """
        # Get suggested filename
        if number is None and doc_type == "upgrade":
            number = self.checker.get_next_upgrade_number()

        filename = self.checker.suggest_name(title, doc_type, number)

        # Validate name
        valid, errors = self.checker.validate_name(filename, doc_type)
        if not valid:
            return False, f"Invalid filename: {'; '.join(errors)}", Path()

        # Check for conflicts
        conflict, conflict_msg = self.checker.check_conflict(filename, doc_type)
        if conflict:
            return False, conflict_msg, Path()

        # Get directory
        config = DOC_CONFIGS[doc_type]
        directory = config["directory"]
        directory.mkdir(parents=True, exist_ok=True)

        filepath = directory / filename

        # Generate content if not provided
        if content is None:
            if doc_type == "upgrade":
                content = self.generator.get_upgrade_template(
                    title,
                    number,
                    kwargs.get("description", ""),
                    kwargs.get("priority", "P1"),
                    kwargs.get("effort", "2-4 weeks"),
                )
            elif doc_type == "research":
                content = self.generator.get_research_template(title, kwargs.get("scope", ""))
            elif doc_type == "insight":
                content = self.generator.get_insight_template(title, kwargs.get("context", ""))
            elif doc_type == "guide":
                content = self.generator.get_guide_template(title, kwargs.get("description", ""))
            else:
                content = f"# {title}\n\nContent goes here."

        # Write file
        filepath.write_text(content)

        return True, f"Created {filepath}", filepath

    def update_index(self, filepath: Path, doc_type: str, description: str = ""):
        """Update the research index with new document."""
        if not RESEARCH_INDEX.exists():
            return

        index_content = RESEARCH_INDEX.read_text()
        filename = filepath.name
        relative_path = filepath.relative_to(RESEARCH_DIR) if filepath.is_relative_to(RESEARCH_DIR) else filepath.name

        # Determine which section to update
        section_markers = {
            "upgrade": "### Upgrade Checklists",
            "research": "### Agent Integration",  # Or appropriate section
            "insight": "### Analysis & Summaries",
            "guide": "### Documentation Management",
        }

        marker = section_markers.get(doc_type)
        if marker and marker in index_content:
            # Find the table in that section and add entry
            # This is a simplified version - could be enhanced
            size_estimate = f"{filepath.stat().st_size // 1024}KB" if filepath.exists() else "NEW"
            new_entry = f"| [{filename}]({relative_path}) | {description or 'NEW'} | {size_estimate} |"

            # Insert after the table header in that section
            # (This is simplified - production version would parse more carefully)
            print(f"Note: Please manually add to index: {new_entry}")

    def list_documents(self, doc_type: str | None = None) -> dict[str, list[str]]:
        """List all documents, optionally filtered by type."""
        if doc_type:
            return {doc_type: self.checker.existing_docs.get(doc_type, [])}
        return self.checker.existing_docs

    def get_next_number(self, doc_type: str = "upgrade") -> int:
        """Get the next available number for a document type."""
        if doc_type == "upgrade":
            return self.checker.get_next_upgrade_number()
        return 1

    def validate_document(self, filepath: Path, doc_type: str) -> tuple[bool, list[str]]:
        """Validate a document has required sections."""
        if not filepath.exists():
            return False, ["File does not exist"]

        content = filepath.read_text()
        config = DOC_CONFIGS.get(doc_type)
        if not config:
            return False, [f"Unknown doc type: {doc_type}"]

        missing = []
        for section in config.get("required_sections", []):
            # Check for section header (## or ###)
            if not re.search(rf"##?\s*.*{re.escape(section)}", content, re.IGNORECASE):
                missing.append(section)

        if missing:
            return False, [f"Missing sections: {', '.join(missing)}"]

        return True, []


# =============================================================================
# CLI Interface
# =============================================================================


def print_help():
    """Print help information."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RESEARCH SAVER - Documentation Tool                        ║
║                      Streamlined templates for AI agents                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUICK CREATE (recommended):                                                 ║
║                                                                              ║
║    python research_saver.py create research "Topic Name"                     ║
║    python research_saver.py create upgrade "Feature Name"                    ║
║    python research_saver.py create guide "Guide Title"                       ║
║    python research_saver.py create insight "Discovery Name"                  ║
║                                                                              ║
║  OTHER COMMANDS:                                                             ║
║                                                                              ║
║    list      List existing documents                                         ║
║    next      Get next available upgrade number                               ║
║    template  Print a template to stdout                                      ║
║    validate  Validate a document has required sections                       ║
║    help      Show this help message                                          ║
║                                                                              ║
║  DOCUMENT TYPES:                                                             ║
║                                                                              ║
║    research  NAME-RESEARCH.md - Research with timestamps                     ║
║    upgrade   UPGRADE-NNN-NAME.md - Implementation checklist                  ║
║    guide     NAME-GUIDE.md - How-to guides                                   ║
║    insight   INSIGHT-DATE-NAME.md - Discoveries/learnings                    ║
║                                                                              ║
║  SLASH COMMANDS (in Claude Code):                                            ║
║                                                                              ║
║    /create-research "Topic"   - Create research doc                          ║
║    /create-upgrade "Title"    - Create upgrade checklist                     ║
║    /create-guide "Title"      - Create how-to guide                          ║
║    /create-insight "Title"    - Create insight doc                           ║
║    /create-doc                - Show all options                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Research Saver - Documentation Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create command - supports both positional and flag-based syntax
    create_parser = subparsers.add_parser("create", help="Create a new document")
    # Positional arguments (simpler syntax)
    create_parser.add_argument(
        "doc_type",
        nargs="?",
        choices=["upgrade", "research", "category", "insight", "guide", "combo"],
        help="Document type (positional)",
    )
    create_parser.add_argument("title", nargs="?", help="Document title (positional)")
    # Flag-based arguments (backwards compatible)
    create_parser.add_argument(
        "--type",
        "-t",
        choices=["upgrade", "research", "category", "insight", "guide", "combo"],
        help="Document type (flag)",
    )
    create_parser.add_argument("--name", "-n", help="Document name/title (flag)")
    create_parser.add_argument("--number", type=int, help="Upgrade number (auto if not specified)")
    create_parser.add_argument("--description", "-d", help="Brief description")
    create_parser.add_argument("--priority", "-p", default="P1", help="Priority (P0/P1/P2)")
    create_parser.add_argument("--effort", "-e", default="", help="Estimated effort")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check filename validity")
    check_parser.add_argument("filename", help="Filename to check")
    check_parser.add_argument("--type", "-t", default="upgrade", help="Document type")

    # List command
    list_parser = subparsers.add_parser("list", help="List existing documents")
    list_parser.add_argument("--type", "-t", help="Filter by document type")

    # Next command
    subparsers.add_parser("next", help="Get next upgrade number")

    # Template command
    template_parser = subparsers.add_parser("template", help="Print template")
    template_parser.add_argument("type", choices=["upgrade", "research", "insight", "guide"], help="Template type")
    template_parser.add_argument("--name", "-n", default="Example", help="Example name")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate document")
    validate_parser.add_argument("filepath", help="Path to document")
    validate_parser.add_argument("--type", "-t", required=True, help="Document type")

    # Help command
    subparsers.add_parser("help", help="Show help")

    args = parser.parse_args()

    saver = ResearchSaver()

    if args.command == "create":
        # Support both positional and flag-based syntax
        doc_type = args.doc_type or args.type
        title = args.title or args.name

        if not doc_type or not title:
            print("❌ Error: Both document type and title are required")
            print("\nUsage:")
            print('  python research_saver.py create research "Topic Name"')
            print('  python research_saver.py create upgrade "Feature Name"')
            print('  python research_saver.py create guide "Guide Title"')
            print('  python research_saver.py create insight "Discovery Name"')
            print('  python research_saver.py create combo "Project Name"')
            sys.exit(1)

        # Handle combo (creates research + insight + upgrade)
        if doc_type == "combo":
            print(f"📦 Creating documentation combo for: {title}")
            created_files = []

            # Create research doc
            success, message, filepath = saver.create_document(
                doc_type="research",
                title=title,
                description=args.description,
                priority=args.priority,
            )
            if success:
                print(f"  ✅ Research: {filepath}")
                created_files.append(("research", filepath))
            else:
                print(f"  ❌ Research: {message}")

            # Create insight doc
            success, message, filepath = saver.create_document(
                doc_type="insight",
                title=title,
                description=args.description,
                priority=args.priority,
            )
            if success:
                print(f"  ✅ Insight: {filepath}")
                created_files.append(("insight", filepath))
            else:
                print(f"  ❌ Insight: {message}")

            # Create upgrade doc
            success, message, filepath = saver.create_document(
                doc_type="upgrade",
                title=title,
                number=args.number,
                description=args.description,
                priority=args.priority,
                effort=args.effort,
            )
            if success:
                print(f"  ✅ Upgrade: {filepath}")
                created_files.append(("upgrade", filepath))
            else:
                print(f"  ❌ Upgrade: {message}")

            if len(created_files) == 3:
                print("\n✅ All 3 documents created successfully!")
            elif len(created_files) > 0:
                print(f"\n⚠️ Created {len(created_files)}/3 documents")
            else:
                print("\n❌ Failed to create documents")
                sys.exit(1)
        else:
            # Single document creation
            success, message, filepath = saver.create_document(
                doc_type=doc_type,
                title=title,
                number=args.number,
                description=args.description,
                priority=args.priority,
                effort=args.effort,
            )
            if success:
                print(f"✅ {message}")
                print(f"   Path: {filepath}")
            else:
                print(f"❌ {message}")
                sys.exit(1)

    elif args.command == "check":
        valid, errors = saver.checker.validate_name(args.filename, args.type)
        conflict, conflict_msg = saver.checker.check_conflict(args.filename, args.type)

        if valid and not conflict:
            print(f"✅ '{args.filename}' is valid and available")
        else:
            if not valid:
                print(f"❌ Invalid: {'; '.join(errors)}")
            if conflict:
                print(f"❌ Conflict: {conflict_msg}")
            sys.exit(1)

    elif args.command == "list":
        docs = saver.list_documents(args.type)
        for doc_type, files in docs.items():
            if files:
                print(f"\n{doc_type.upper()} ({len(files)} files):")
                for f in sorted(files):
                    print(f"  - {f}")

    elif args.command == "next":
        next_num = saver.get_next_number()
        print(f"Next upgrade number: {next_num}")
        print(f"Suggested filename: UPGRADE-{next_num:03d}-YOUR-NAME.md")

    elif args.command == "template":
        gen = TemplateGenerator()
        if args.type == "upgrade":
            print(gen.get_upgrade_template(args.name, 999))
        elif args.type == "research":
            print(gen.get_research_template(args.name))
        elif args.type == "insight":
            print(gen.get_insight_template(args.name))
        elif args.type == "guide":
            print(gen.get_guide_template(args.name))

    elif args.command == "validate":
        valid, errors = saver.validate_document(Path(args.filepath), args.type)
        if valid:
            print("✅ Document is valid")
        else:
            print(f"❌ Validation failed: {'; '.join(errors)}")
            sys.exit(1)

    elif args.command == "help" or args.command is None:
        print_help()

    else:
        print_help()


if __name__ == "__main__":
    main()
