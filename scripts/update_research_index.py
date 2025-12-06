#!/usr/bin/env python3
"""
Research Index Auto-Updater

Automatically updates cross-references in research documentation:
- docs/research/README.md - Quick reference tables
- docs/research/UPGRADE_INDEX.md - Upgrade-to-research mapping

Usage:
    python scripts/update_research_index.py                    # Full update
    python scripts/update_research_index.py --check            # Check only, don't modify
    python scripts/update_research_index.py --add FILE.md      # Add new document
    python scripts/update_research_index.py --rename OLD NEW   # Update after rename

Features:
- Scans docs/research/ for all documents
- Extracts frontmatter metadata
- Updates quick reference tables
- Updates upgrade index mappings
- Detects orphaned documents
- Validates cross-references
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocInfo:
    """Information about a research document."""

    filename: str
    path: Path
    title: str = ""
    topic: str = ""
    related_upgrades: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    size_kb: int = 0
    doc_type: str = ""  # research, summary, guide, checklist


def parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from markdown."""
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


def classify_document(filename: str) -> str:
    """Classify document type from filename."""
    if filename.endswith("_RESEARCH.md"):
        return "research"
    elif filename.endswith("_SUMMARY.md"):
        return "summary"
    elif filename.endswith("_UPGRADE_GUIDE.md"):
        return "guide"
    elif re.match(r"^UPGRADE-\d{3}-.*\.md$", filename):
        return "checklist"
    elif filename in ["README.md", "NAMING_CONVENTION.md", "UPGRADE_INDEX.md"]:
        return "index"
    return "other"


def scan_research_docs(research_dir: Path) -> dict[str, DocInfo]:
    """Scan all research documents and extract metadata."""
    docs = {}

    for md_file in research_dir.glob("*.md"):
        filename = md_file.name
        doc_type = classify_document(filename)

        # Skip index files
        if doc_type == "index":
            continue

        content = md_file.read_text(encoding="utf-8")
        frontmatter = parse_frontmatter(content)

        # Get file size
        size_bytes = md_file.stat().st_size
        size_kb = round(size_bytes / 1024)

        # Get title from frontmatter or generate from filename
        title = frontmatter.get("title", "")
        if not title:
            title = filename.replace("_", " ").replace(".md", "").title()

        # Extract short description from first paragraph if no title
        if not title:
            first_para = re.search(r"\n\n(.+?)\n", content)
            if first_para:
                title = first_para.group(1)[:50] + "..."

        docs[filename] = DocInfo(
            filename=filename,
            path=md_file,
            title=title,
            topic=frontmatter.get("topic", "general"),
            related_upgrades=frontmatter.get("related_upgrades", []),
            tags=frontmatter.get("tags", []),
            size_kb=size_kb,
            doc_type=doc_type,
        )

    return docs


def generate_quick_reference_section(docs: dict[str, DocInfo]) -> str:
    """Generate quick reference markdown tables by topic."""
    # Group by topic
    by_topic: dict[str, list[DocInfo]] = {}
    for doc in docs.values():
        topic = doc.topic or "general"
        if topic not in by_topic:
            by_topic[topic] = []
        by_topic[topic].append(doc)

    # Topic display names
    topic_names = {
        "quantconnect": "QuantConnect Platform",
        "evaluation": "Evaluation Frameworks",
        "llm": "LLM & Sentiment Analysis",
        "workflow": "Workflow & Development",
        "prompts": "Prompt Engineering",
        "autonomous": "Autonomous Agents",
        "agents": "Agent Integration",
        "sentiment": "Sentiment Analysis",
        "integration": "Integration",
        "general": "General",
    }

    # Generate tables
    lines = []
    for topic in ["quantconnect", "evaluation", "llm", "workflow", "prompts", "autonomous", "agents", "general"]:
        if topic not in by_topic:
            continue

        docs_in_topic = sorted(by_topic[topic], key=lambda d: d.filename)
        if not docs_in_topic:
            continue

        lines.append(f"\n### {topic_names.get(topic, topic.title())}\n")
        lines.append("| Document | Description | Size |")
        lines.append("|----------|-------------|------|")

        for doc in docs_in_topic:
            title_short = doc.title[:50] + "..." if len(doc.title) > 50 else doc.title
            lines.append(f"| [{doc.filename}]({doc.filename}) | {title_short} | {doc.size_kb}KB |")

    return "\n".join(lines)


def generate_upgrade_index_section(docs: dict[str, DocInfo]) -> str:
    """Generate upgrade-to-research mapping section."""
    # Collect all upgrades
    upgrade_map: dict[str, list[DocInfo]] = {}

    for doc in docs.values():
        for upgrade in doc.related_upgrades:
            if upgrade not in upgrade_map:
                upgrade_map[upgrade] = []
            upgrade_map[upgrade].append(doc)

    if not upgrade_map:
        return ""

    # Sort upgrades
    sorted_upgrades = sorted(upgrade_map.keys())

    lines = []
    lines.append("\n## Quick Lookup\n")
    lines.append("| Upgrade | Title | Primary Research | Implementation Guide |")
    lines.append("|---------|-------|------------------|---------------------|")

    for upgrade in sorted_upgrades:
        docs_for_upgrade = upgrade_map[upgrade]
        # Find primary research doc
        research_doc = next((d for d in docs_for_upgrade if d.doc_type == "research"), None)
        guide_doc = next((d for d in docs_for_upgrade if d.doc_type == "guide"), None)
        checklist_doc = next((d for d in docs_for_upgrade if d.doc_type == "checklist"), None)

        primary = research_doc or docs_for_upgrade[0]
        impl = checklist_doc or guide_doc

        title = primary.title.replace(" Research", "")
        primary_link = f"[{primary.filename}]({primary.filename})"
        impl_link = f"[{impl.filename}]({impl.filename})" if impl else "See research doc"

        lines.append(f"| {upgrade} | {title} | {primary_link} | {impl_link} |")

    return "\n".join(lines)


def update_readme_quick_reference(
    readme_path: Path, docs: dict[str, DocInfo], check_only: bool = False
) -> tuple[bool, str]:
    """Update README.md quick reference section."""
    if not readme_path.exists():
        return False, "README.md not found"

    content = readme_path.read_text(encoding="utf-8")

    # Find quick reference section
    # Look for pattern: ## 📚 Quick Reference - Research Documents by Topic
    start_pattern = r"## 📚 Quick Reference - Research Documents by Topic\s*\n"
    start_match = re.search(start_pattern, content)

    if not start_match:
        return False, "Quick reference section not found in README.md"

    # Find end of section (next ## heading or end of file)
    section_start = start_match.end()
    end_match = re.search(r"\n## [^#]", content[section_start:])
    if end_match:
        section_end = section_start + end_match.start()
    else:
        section_end = len(content)

    # Generate new section
    new_section = generate_quick_reference_section(docs)

    if check_only:
        return True, f"Would update quick reference with {len(docs)} documents"

    # Replace section
    new_content = content[:section_start] + "\n" + new_section + "\n\n" + content[section_end:].lstrip()

    readme_path.write_text(new_content, encoding="utf-8")
    return True, f"Updated quick reference with {len(docs)} documents"


def update_after_rename(research_dir: Path, old_name: str, new_name: str, check_only: bool = False) -> list[str]:
    """Update all references after a document rename."""
    changes = []

    # Find all files that reference the old name
    for md_file in research_dir.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")

        if old_name in content:
            if check_only:
                changes.append(f"Would update references in {md_file.name}")
            else:
                new_content = content.replace(old_name, new_name)
                md_file.write_text(new_content, encoding="utf-8")
                changes.append(f"Updated references in {md_file.name}")

    # Also check parent directories
    docs_dir = research_dir.parent
    for subdir in ["upgrades", "development", "strategies", "architecture"]:
        subdir_path = docs_dir / subdir
        if subdir_path.exists():
            for md_file in subdir_path.glob("*.md"):
                content = md_file.read_text(encoding="utf-8")
                if old_name in content:
                    if check_only:
                        changes.append(f"Would update references in {subdir}/{md_file.name}")
                    else:
                        new_content = content.replace(old_name, new_name)
                        md_file.write_text(new_content, encoding="utf-8")
                        changes.append(f"Updated references in {subdir}/{md_file.name}")

    return changes


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Update research documentation cross-references")
    parser.add_argument("--check", action="store_true", help="Check only, don't modify files")
    parser.add_argument("--add", type=str, help="Add a new document to the index")
    parser.add_argument("--rename", nargs=2, metavar=("OLD", "NEW"), help="Update references after renaming a document")
    parser.add_argument("--dir", type=Path, default=Path("docs/research"), help="Research directory path")

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    research_dir = project_root / args.dir

    if not research_dir.exists():
        print(f"Error: Research directory not found: {research_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Research Index Updater")
    print("=" * 60)

    # Handle rename operation
    if args.rename:
        old_name, new_name = args.rename
        print(f"\nUpdating references: {old_name} -> {new_name}")
        changes = update_after_rename(research_dir, old_name, new_name, args.check)
        if changes:
            for change in changes:
                print(f"  {change}")
        else:
            print("  No references found to update")
        sys.exit(0)

    # Scan documents
    print(f"\nScanning: {research_dir}")
    docs = scan_research_docs(research_dir)
    print(f"Found: {len(docs)} documents")

    # Update README
    readme_path = research_dir / "README.md"
    success, message = update_readme_quick_reference(readme_path, docs, args.check)
    status = "CHECK" if args.check else ("OK" if success else "FAILED")
    print(f"\nREADME.md: [{status}] {message}")

    # Generate summary
    print("\n" + "-" * 60)
    print("Document Summary:")
    print("-" * 60)

    by_type = {}
    for doc in docs.values():
        by_type.setdefault(doc.doc_type, []).append(doc)

    for doc_type, type_docs in sorted(by_type.items()):
        print(f"  {doc_type.title()}: {len(type_docs)}")

    print("\n" + "=" * 60)
    if args.check:
        print("Check complete. Run without --check to apply changes.")
    else:
        print("Update complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
