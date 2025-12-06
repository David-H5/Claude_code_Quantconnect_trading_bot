#!/usr/bin/env python3
"""
Cross-Reference Updater Hook

Automatically maintains cross-references between:
- Research docs and implementation files
- Research docs and progress file
- Research docs and CLAUDE.md

Triggers on: PostToolUse for Write/Edit
Purpose: Keep documentation cross-references up to date
"""

import json
import sys
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Mapping of research doc patterns to implementation directories
RESEARCH_TO_IMPL = {
    "OBSERVABILITY": ["observability/", "llm/agents/observability"],
    "FAULT-TOLERANCE": ["models/circuit_breaker", "execution/"],
    "MEMORY-MANAGEMENT": ["llm/memory/", "utils/memory_"],
    "SAFETY-GUARDRAILS": ["models/risk_manager", "models/circuit_breaker"],
    "COST-OPTIMIZATION": ["llm/cost_", "llm/cache_"],
    "STATE-PERSISTENCE": ["llm/checkpointer", "models/state_"],
    "TESTING-SIMULATION": ["tests/", "evaluation/"],
    "SELF-IMPROVEMENT": ["llm/self_evolving", "llm/prompt_optimizer"],
    "WORKSPACE-MANAGEMENT": ["llm/agents/workspace_", "utils/workspace"],
    "OVERNIGHT-SESSIONS": ["scripts/run_overnight", "scripts/watchdog"],
    "CLAUDE-CODE": [".claude/", "scripts/"],
}


def find_related_research_docs(impl_file: str) -> list[Path]:
    """Find research docs related to an implementation file."""
    related = []
    research_dir = PROJECT_ROOT / "docs" / "research"

    if not research_dir.exists():
        return related

    impl_lower = impl_file.lower()

    for pattern, impl_patterns in RESEARCH_TO_IMPL.items():
        for impl_pattern in impl_patterns:
            if impl_pattern.lower() in impl_lower:
                # Find matching research docs
                for doc in research_dir.glob(f"*{pattern}*.md"):
                    if doc not in related:
                        related.append(doc)

    return related


def find_related_impl_files(research_doc: Path) -> list[Path]:
    """Find implementation files related to a research doc.

    Uses efficient directory-based search instead of rglob to avoid
    timeout issues on large codebases.
    """
    related = []
    doc_name = research_doc.stem.upper()
    max_files = 10

    for pattern, impl_patterns in RESEARCH_TO_IMPL.items():
        if pattern in doc_name:
            for impl_pattern in impl_patterns:
                # Extract directory and file prefix
                if "/" in impl_pattern:
                    dir_part, file_prefix = impl_pattern.rsplit("/", 1)
                    search_dir = PROJECT_ROOT / dir_part
                else:
                    search_dir = PROJECT_ROOT
                    file_prefix = impl_pattern

                # Only search if directory exists
                if not search_dir.exists():
                    continue

                # Use non-recursive glob (fast)
                try:
                    for f in search_dir.glob(f"{file_prefix}*.py"):
                        if f.is_file() and f not in related:
                            related.append(f)
                            if len(related) >= max_files:
                                return related
                except (PermissionError, OSError):
                    continue

    return related


def suggest_cross_references(file_path: str) -> str:
    """Generate cross-reference suggestions based on file path."""
    path = Path(file_path)
    suggestions = []

    # Check if it's an implementation file
    if path.suffix == ".py":
        related_docs = find_related_research_docs(file_path)
        if related_docs:
            suggestions.append("\n## Cross-Reference Suggestion\n")
            suggestions.append(f"Add docstring reference to `{path.name}`:\n")
            suggestions.append("```python")
            suggestions.append('"""')
            suggestions.append(f"Module: {path.stem}")
            suggestions.append("")
            suggestions.append("Research References:")
            for doc in related_docs[:3]:
                rel_path = doc.relative_to(PROJECT_ROOT)
                suggestions.append(f"    See: {rel_path}")
            suggestions.append('"""')
            suggestions.append("```")

    # Check if it's a research doc
    elif "docs/research" in file_path and file_path.endswith(".md"):
        doc_path = PROJECT_ROOT / file_path.lstrip("/")
        if doc_path.exists():
            related_files = find_related_impl_files(doc_path)
            if related_files:
                suggestions.append("\n## Implementation Files to Update\n")
                suggestions.append("Add research doc reference to these files:\n")
                for f in related_files[:5]:
                    rel_path = f.relative_to(PROJECT_ROOT)
                    suggestions.append(f"- `{rel_path}`")
                suggestions.append("")
                suggestions.append("**Docstring template**:")
                suggestions.append("```python")
                suggestions.append(f'See: {file_path.split("docs/research/")[-1]}')
                suggestions.append("```")

    return "\n".join(suggestions)


def update_research_index(research_doc: Path) -> str:
    """Suggest updates to research index."""
    index_path = PROJECT_ROOT / "docs" / "research" / "README.md"

    if not index_path.exists():
        return ""

    content = index_path.read_text()
    doc_name = research_doc.name

    if doc_name not in content:
        return f"""
## Research Index Update Required

Add to `docs/research/README.md`:

```markdown
| [{research_doc.stem}]({doc_name}) | [Topic] | [Status] |
```

Run: `python scripts/update_research_index.py`
"""
    return ""


def main():
    """Main hook entry point."""
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Only process Write/Edit
    if tool_name not in ["Write", "Edit"]:
        sys.exit(0)

    file_path = tool_input.get("file_path", "")
    if not file_path:
        sys.exit(0)

    # Skip if not a relevant file
    is_py_file = file_path.endswith(".py")
    is_research_doc = "docs/research/" in file_path and file_path.endswith(".md")

    if not is_py_file and not is_research_doc:
        sys.exit(0)

    # Generate suggestions
    suggestions = suggest_cross_references(file_path)

    # For research docs, also suggest index update
    if is_research_doc:
        doc_path = PROJECT_ROOT / file_path.lstrip("/")
        if doc_path.exists():
            index_suggestion = update_research_index(doc_path)
            suggestions += index_suggestion

    if suggestions.strip():
        print(suggestions)

    sys.exit(0)


if __name__ == "__main__":
    main()
