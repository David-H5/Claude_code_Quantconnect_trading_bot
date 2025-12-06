#!/usr/bin/env python3
"""Template Detection Hook for Claude Code.

Detects natural language requests for documentation templates and suggests
the appropriate template type. Also detects update requests and finds
relevant existing documents. Runs on UserPromptSubmit.

Usage in settings.json:
{
  "hooks": {
    "UserPromptSubmit": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "python3 .claude/hooks/template_detector.py \"$PROMPT\""
      }]
    }]
  }
}
"""

import re
import sys
from datetime import datetime
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESEARCH_DIR = PROJECT_ROOT / "docs" / "research"
INSIGHTS_DIR = PROJECT_ROOT / "docs" / "insights"
GUIDES_DIR = PROJECT_ROOT / "docs" / "guides"

# Template detection patterns
TEMPLATE_PATTERNS = {
    "research": {
        "keywords": [
            r"create\s+research\s+doc",
            r"research\s+document",
            r"new\s+research",
            r"start\s+research",
            r"document\s+research",
            r"research\s+template",
            r"create\s+.*research",
        ],
        "command": "/create-research",
        "description": "Research documentation with timestamped sources",
    },
    "upgrade": {
        "keywords": [
            r"create\s+upgrade\s+guide",
            r"upgrade\s+document",
            r"new\s+upgrade",
            r"implementation\s+checklist",
            r"upgrade\s+template",
            r"create\s+upgrade",
            r"upgrade\s+plan",
            r"implementation\s+plan",
        ],
        "command": "/create-upgrade",
        "description": "Upgrade checklist with phases and tasks",
    },
    "guide": {
        "keywords": [
            r"create\s+guide",
            r"how-?to\s+guide",
            r"tutorial\s+document",
            r"new\s+guide",
            r"guide\s+template",
            r"create\s+tutorial",
            r"documentation\s+guide",
        ],
        "command": "/create-guide",
        "description": "How-to guide with steps and examples",
    },
    "insight": {
        "keywords": [
            r"create\s+insight",
            r"document\s+insight",
            r"insight\s+document",
            r"new\s+insight",
            r"insight\s+template",
            r"discovery\s+document",
            r"document\s+discovery",
        ],
        "command": "/create-insight",
        "description": "Insight documentation with evidence",
    },
    "combo": {
        "keywords": [
            r"create\s+documents",
            r"make\s+documents",
            r"create\s+all\s+docs",
            r"full\s+documentation",
            r"complete\s+documentation",
            r"document\s+this\s+project",
            r"document\s+the\s+project",
            r"project\s+documentation",
            r"create\s+project\s+docs",
            r"make\s+project\s+docs",
            r"generate\s+documentation",
            r"documentation\s+combo",
            r"docs\s+combo",
        ],
        "command": "/create-docs-combo",
        "description": "Full documentation set (Research + Insight + Upgrade)",
        "is_combo": True,
    },
}

# Update detection patterns
UPDATE_PATTERNS = {
    "research": {
        "keywords": [
            r"update\s+research\s+doc",
            r"update\s+research",
            r"modify\s+research",
            r"add\s+to\s+research",
            r"edit\s+research",
        ],
        "directory": RESEARCH_DIR,
        "file_pattern": r".*-RESEARCH\.md$",
        "description": "research documentation",
    },
    "insight": {
        "keywords": [
            r"update\s+insight\s+doc",
            r"update\s+insight",
            r"modify\s+insight",
            r"add\s+to\s+insight",
            r"edit\s+insight",
        ],
        "directory": INSIGHTS_DIR,
        "file_pattern": r"INSIGHT-.*\.md$",
        "description": "insight documentation",
    },
    "upgrade": {
        "keywords": [
            r"update\s+upgrade\s+guide",
            r"update\s+upgrade",
            r"modify\s+upgrade",
            r"add\s+to\s+upgrade",
            r"edit\s+upgrade",
        ],
        "directory": RESEARCH_DIR,
        "file_pattern": r"UPGRADE-\d+.*\.md$",
        "description": "upgrade guide",
    },
    "guide": {
        "keywords": [
            r"update\s+guide",
            r"update\s+guides",
            r"modify\s+guide",
            r"add\s+to\s+guide",
            r"edit\s+guide",
        ],
        "directory": GUIDES_DIR,
        "file_pattern": r".*-GUIDE\.md$",
        "description": "how-to guide",
    },
    "documents": {
        "keywords": [
            r"update\s+document",
            r"update\s+documents",
            r"update\s+docs",
            r"modify\s+document",
            r"edit\s+document",
        ],
        "is_multi": True,
        "description": "all related documents",
    },
}


def find_recent_documents(doc_type: str, limit: int = 5) -> list[dict]:
    """Find recent documents of a specific type.

    Returns list of dicts with 'path', 'name', 'modified' keys.
    """
    config = UPDATE_PATTERNS.get(doc_type, {})
    directory = config.get("directory")
    pattern = config.get("file_pattern")

    if not directory or not pattern or not directory.exists():
        return []

    docs = []
    regex = re.compile(pattern)

    for file in directory.glob("*.md"):
        if regex.match(file.name):
            mtime = file.stat().st_mtime
            docs.append(
                {
                    "path": file,
                    "name": file.name,
                    "modified": datetime.fromtimestamp(mtime),
                }
            )

    # Sort by modification time (most recent first)
    docs.sort(key=lambda x: x["modified"], reverse=True)
    return docs[:limit]


def find_all_recent_documents(limit_per_type: int = 3) -> dict[str, list[dict]]:
    """Find recent documents across all types."""
    all_docs = {}
    for doc_type in ["research", "insight", "upgrade", "guide"]:
        docs = find_recent_documents(doc_type, limit_per_type)
        if docs:
            all_docs[doc_type] = docs
    return all_docs


def detect_update_request(prompt: str) -> tuple[str, dict, list[dict]] | None:
    """Detect if the prompt is requesting a document update.

    Returns:
        Tuple of (doc_type, config, found_docs) if detected, None otherwise
    """
    prompt_lower = prompt.lower()

    for doc_type, config in UPDATE_PATTERNS.items():
        for pattern in config["keywords"]:
            if re.search(pattern, prompt_lower):
                if config.get("is_multi"):
                    # Find docs across all types
                    all_docs = find_all_recent_documents(3)
                    return doc_type, config, all_docs
                else:
                    # Find docs of specific type
                    docs = find_recent_documents(doc_type, 5)
                    return doc_type, config, docs

    return None


def format_update_suggestion(doc_type: str, config: dict, docs) -> str:
    """Format an update suggestion message with found documents."""
    if config.get("is_multi"):
        # Multi-document update
        lines = ["📝 **Update Documents Detected**\n"]
        lines.append("Found recent documents to update:\n")

        if not docs:
            lines.append("_No recent documents found._\n")
        else:
            for dtype, doc_list in docs.items():
                lines.append(f"\n**{dtype.title()}** ({len(doc_list)} recent):")
                for doc in doc_list[:3]:
                    rel_path = doc["path"].relative_to(PROJECT_ROOT)
                    age = (datetime.now() - doc["modified"]).days
                    age_str = f"{age}d ago" if age > 0 else "today"
                    lines.append(f"  - [{doc['name']}]({rel_path}) ({age_str})")

        lines.append("\n\n**To update**, read the document first, then edit with new information.")
        lines.append("Ensure updates follow the template structure (timestamped sources, etc.).")
        return "\n".join(lines)

    # Single type update
    lines = [f"📝 **Update {doc_type.title()} Detected**\n"]
    lines.append(f"Found recent {config['description']}:\n")

    if not docs:
        lines.append(f"_No {config['description']} files found._\n")
        lines.append(f'\nCreate one with: `/create-{doc_type} "Title"`')
    else:
        for doc in docs:
            rel_path = doc["path"].relative_to(PROJECT_ROOT)
            age = (datetime.now() - doc["modified"]).days
            age_str = f"{age}d ago" if age > 0 else "today"
            lines.append(f"  - [{doc['name']}]({rel_path}) ({age_str})")

        lines.append("\n\n**To update:**")
        lines.append("1. Read the target document")
        lines.append("2. Add new information following the template structure")
        lines.append("3. Update timestamps and changelog")

        if doc_type == "research":
            lines.append("\n**Research updates must include:**")
            lines.append("- Search timestamp: `YYYY-MM-DD at HH:MM TZ`")
            lines.append("- Source dates: `Published: YYYY-MM`")
        elif doc_type == "upgrade":
            lines.append("\n**Upgrade updates:**")
            lines.append("- Update task status: ⬜→🔄→✅")
            lines.append("- Update progress percentage")
            lines.append("- Add changelog entry")

    return "\n".join(lines)


def detect_template_request(prompt: str) -> tuple[str, dict] | None:
    """Detect if the prompt is requesting a template.

    Returns:
        Tuple of (template_type, config) if detected, None otherwise
    """
    prompt_lower = prompt.lower()

    for template_type, config in TEMPLATE_PATTERNS.items():
        for pattern in config["keywords"]:
            if re.search(pattern, prompt_lower):
                return template_type, config

    return None


def format_suggestion(template_type: str, config: dict) -> str:
    """Format a template suggestion message."""
    if config.get("is_combo"):
        return f"""📦 **Combo Documentation Detected**

Use `{config['command']} "<project-name>"` to create a full documentation set:
- **Research doc** - Background research with timestamped sources
- **Insight doc** - Key discoveries and learnings
- **Upgrade guide** - Implementation checklist with phases

**Quick command:**
```bash
python3 .claude/hooks/research_saver.py create combo "Project Name"
```

This creates 3 linked documents for comprehensive project documentation.

**Individual docs:**
- `/create-research "Topic"` - Just research
- `/create-insight "Discovery"` - Just insight
- `/create-upgrade "Feature"` - Just upgrade checklist"""

    return f"""📋 **Template Detected**: {template_type.title()}

Use `{config['command']} "<title>"` to create a {config['description']}.

**Quick commands:**
- `/create-research "Topic Name"` - Research doc
- `/create-upgrade "Feature Name"` - Upgrade checklist
- `/create-guide "Guide Title"` - How-to guide
- `/create-insight "Discovery Name"` - Insight doc
- `/create-docs-combo "Project"` - Full set (Research + Insight + Upgrade)

Or use the CLI directly:
```bash
python3 .claude/hooks/research_saver.py create {template_type} "Title"
```"""


def main():
    """Main entry point for the hook."""
    if len(sys.argv) < 2:
        # No prompt provided, exit silently
        sys.exit(0)

    prompt = sys.argv[1]

    # Check for update request first (more specific)
    update_result = detect_update_request(prompt)
    if update_result:
        doc_type, config, docs = update_result
        print(format_update_suggestion(doc_type, config, docs))
        sys.exit(0)

    # Then check for create request
    result = detect_template_request(prompt)
    if result:
        template_type, config = result
        print(format_suggestion(template_type, config))

    # Always exit 0 to not block the prompt
    sys.exit(0)


if __name__ == "__main__":
    main()
