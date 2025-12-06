#!/usr/bin/env python3
"""
Research Document Validation Script

Validates research documents against naming conventions and cross-referencing rules.
Part of the automated research documentation system.

Usage:
    python scripts/validate_research_docs.py [--fix] [--verbose]

Options:
    --fix       Attempt to auto-fix issues where possible
    --verbose   Show detailed output
    --json      Output results as JSON for automation

Exit codes:
    0 - All checks passed
    1 - Validation errors found
    2 - Script error
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar


@dataclass
class ValidationResult:
    """Result of a validation check."""

    file: str
    check: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info
    suggestion: str | None = None


@dataclass
class DocumentInfo:
    """Information extracted from a research document."""

    path: Path
    filename: str
    has_frontmatter: bool = False
    frontmatter: dict = field(default_factory=dict)
    title: str | None = None
    topic: str | None = None
    related_upgrades: list[str] = field(default_factory=list)
    related_docs: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    internal_links: list[str] = field(default_factory=list)
    word_count: int = 0


class ResearchDocValidator:
    """Validates research documentation against naming conventions."""

    # Valid naming patterns
    VALID_PATTERNS: ClassVar[list[str]] = [
        r"^[A-Z][A-Z0-9_]+_RESEARCH\.md$",  # TOPIC_RESEARCH.md
        r"^[A-Z][A-Z0-9_]+_SUMMARY\.md$",  # TOPIC_SUMMARY.md
        r"^[A-Z][A-Z0-9_]+_UPGRADE_GUIDE\.md$",  # TOPIC_UPGRADE_GUIDE.md
        r"^UPGRADE-\d{3}-[A-Z][A-Z0-9-]+\.md$",  # UPGRADE-NNN-TOPIC.md
        r"^UPGRADE-\d{3}\.\d+-[A-Z][A-Z0-9-]+\.md$",  # UPGRADE-NNN.N-TOPIC.md (sub-upgrade)
        r"^UPGRADE-\d{3}-CAT\d+-[A-Z][A-Z0-9-]+-RESEARCH\.md$",  # UPGRADE-NNN-CATN-TOPIC-RESEARCH.md (category)
        r"^UPGRADE-\d{3}-SPRINT\d+(\.\d+)?-[A-Z][A-Z0-9-]+\.md$",  # UPGRADE-NNN-SPRINTN.N-TOPIC.md (sprint)
        r"^README\.md$",  # Index file
        r"^NAMING_CONVENTION\.md$",  # This convention doc
        r"^UPGRADE_INDEX\.md$",  # Upgrade index
        r"^[A-Z][A-Z0-9_\-]+\s*\(\d+\)\.md$",  # Legacy files with (1) suffix
    ]

    # Required frontmatter fields for research documents
    REQUIRED_FRONTMATTER: ClassVar[list[str]] = ["title", "topic", "tags", "created"]

    # Valid topic categories
    VALID_TOPICS: ClassVar[list[str]] = [
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

    def __init__(self, research_dir: Path, verbose: bool = False):
        """Initialize validator with research directory path."""
        self.research_dir = research_dir
        self.verbose = verbose
        self.results: list[ValidationResult] = []
        self.documents: dict[str, DocumentInfo] = {}

    def log(self, message: str) -> None:
        """Log verbose message."""
        if self.verbose:
            print(f"  [DEBUG] {message}")

    def validate_all(self) -> list[ValidationResult]:
        """Run all validation checks."""
        self.results = []

        # First pass: collect document info
        self._collect_documents()

        # Run validation checks
        self._check_naming_conventions()
        self._check_frontmatter()
        self._check_cross_references()
        self._check_orphaned_documents()
        self._check_upgrade_index_coverage()
        self._check_content_quality()
        self._check_duplicate_links()

        return self.results

    def _collect_documents(self) -> None:
        """Collect information from all research documents."""
        self.log(f"Scanning {self.research_dir}")

        for md_file in self.research_dir.glob("*.md"):
            doc_info = self._parse_document(md_file)
            self.documents[md_file.name] = doc_info
            self.log(f"Found: {md_file.name}")

    def _parse_document(self, path: Path) -> DocumentInfo:
        """Parse a markdown document and extract metadata."""
        content = path.read_text(encoding="utf-8")
        doc = DocumentInfo(path=path, filename=path.name)

        # Extract frontmatter if present
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if frontmatter_match:
            doc.has_frontmatter = True
            doc.frontmatter = self._parse_yaml_frontmatter(frontmatter_match.group(1))
            doc.title = doc.frontmatter.get("title")
            doc.topic = doc.frontmatter.get("topic")
            doc.related_upgrades = doc.frontmatter.get("related_upgrades", [])
            doc.related_docs = doc.frontmatter.get("related_docs", [])
            doc.tags = doc.frontmatter.get("tags", [])

        # Extract internal links
        doc.internal_links = re.findall(r"\[.*?\]\(([^)]+\.md)\)", content)

        # Count words (rough estimate)
        doc.word_count = len(content.split())

        return doc

    def _parse_yaml_frontmatter(self, yaml_str: str) -> dict:
        """Simple YAML frontmatter parser (no external deps)."""
        result = {}
        current_key = None
        current_list = []

        for line in yaml_str.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Check for key: value
            if ":" in line and not line.startswith("-"):
                if current_key and current_list:
                    result[current_key] = current_list
                    current_list = []

                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()

                if value.startswith("[") and value.endswith("]"):
                    # Inline list
                    items = value[1:-1].split(",")
                    result[key] = [item.strip().strip("'\"") for item in items if item.strip()]
                elif value:
                    result[key] = value.strip("'\"")
                else:
                    current_key = key
            elif line.startswith("-") and current_key:
                # List item
                item = line[1:].strip().strip("'\"")
                current_list.append(item)

        if current_key and current_list:
            result[current_key] = current_list

        return result

    def _check_naming_conventions(self) -> None:
        """Check that all documents follow naming conventions."""
        for filename, _doc in self.documents.items():
            is_valid = any(re.match(pattern, filename) for pattern in self.VALID_PATTERNS)

            if not is_valid:
                # Check if it's close to a valid pattern
                suggestion = self._suggest_name(filename)
                self.results.append(
                    ValidationResult(
                        file=filename,
                        check="naming_convention",
                        passed=False,
                        message=f"Filename '{filename}' does not match naming convention",
                        suggestion=suggestion,
                    )
                )
            else:
                self.results.append(
                    ValidationResult(
                        file=filename,
                        check="naming_convention",
                        passed=True,
                        message="Naming convention OK",
                        severity="info",
                    )
                )

    def _suggest_name(self, filename: str) -> str | None:
        """Suggest a corrected filename."""
        # Remove common issues
        name = filename.replace(".md", "")

        # If it has a date suffix, suggest removing it
        date_match = re.search(r"_?(DEC|NOV|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT)\d{4}$", name, re.IGNORECASE)
        if date_match:
            name = name[: date_match.start()]
            return f"{name.upper()}.md (move date to frontmatter)"

        # If it starts with UPGRADE_ but isn't a checklist
        if name.startswith("UPGRADE_") and "RESEARCH" in name:
            # Extract topic
            topic = name.replace("UPGRADE_", "").replace("_RESEARCH", "")
            topic = re.sub(r"^\d+_?", "", topic)  # Remove leading numbers
            return f"{topic}_RESEARCH.md (add upgrade number to frontmatter)"

        return None

    def _check_frontmatter(self) -> None:
        """Check frontmatter presence and required fields."""
        for filename, doc in self.documents.items():
            # Skip index and convention files
            if filename in ["README.md", "NAMING_CONVENTION.md", "UPGRADE_INDEX.md"]:
                continue

            # Only check research documents
            if not filename.endswith("_RESEARCH.md"):
                continue

            if not doc.has_frontmatter:
                self.results.append(
                    ValidationResult(
                        file=filename,
                        check="frontmatter_presence",
                        passed=False,
                        message="Missing YAML frontmatter",
                        severity="warning",
                        suggestion="Add frontmatter with title, topic, tags, created fields",
                    )
                )
            else:
                # Check required fields
                missing = []
                for field in self.REQUIRED_FRONTMATTER:
                    if field not in doc.frontmatter:
                        missing.append(field)

                if missing:
                    self.results.append(
                        ValidationResult(
                            file=filename,
                            check="frontmatter_fields",
                            passed=False,
                            message=f"Missing frontmatter fields: {', '.join(missing)}",
                            severity="warning",
                        )
                    )

                # Validate topic
                if doc.topic and doc.topic.lower() not in self.VALID_TOPICS:
                    self.results.append(
                        ValidationResult(
                            file=filename,
                            check="frontmatter_topic",
                            passed=False,
                            message=f"Invalid topic '{doc.topic}'. Valid: {', '.join(self.VALID_TOPICS)}",
                            severity="warning",
                        )
                    )

    def _check_cross_references(self) -> None:
        """Check that all internal links resolve."""
        for filename, doc in self.documents.items():
            for link in doc.internal_links:
                # Handle relative paths by properly resolving from document location
                if link.startswith("../"):
                    # Count parent directory traversals
                    remaining = link
                    base_path = self.research_dir
                    while remaining.startswith("../"):
                        base_path = base_path.parent
                        remaining = remaining[3:]  # Remove "../"
                    link_path = base_path / remaining
                elif link.startswith("docs/"):
                    # Absolute from project root
                    link_path = self.research_dir.parent.parent / link
                else:
                    link_path = self.research_dir / link

                # Normalize and check
                link_path = link_path.resolve()
                if not link_path.exists():
                    self.results.append(
                        ValidationResult(
                            file=filename,
                            check="cross_reference",
                            passed=False,
                            message=f"Broken link: {link}",
                            severity="error",
                        )
                    )

    def _check_orphaned_documents(self) -> None:
        """Check for documents with no incoming links."""
        # Collect all referenced documents
        referenced: set[str] = set()
        for doc in self.documents.values():
            for link in doc.internal_links:
                # Extract just the filename
                link_name = Path(link).name
                referenced.add(link_name)

        # Check for orphans
        for filename in self.documents:
            # Skip index files
            if filename in ["README.md", "NAMING_CONVENTION.md", "UPGRADE_INDEX.md"]:
                continue

            if filename not in referenced:
                self.results.append(
                    ValidationResult(
                        file=filename,
                        check="orphaned_document",
                        passed=False,
                        message="Document has no incoming links (orphaned)",
                        severity="warning",
                        suggestion="Add to README.md quick reference or link from related docs",
                    )
                )

    def _check_upgrade_index_coverage(self) -> None:
        """Check that upgrade index covers all upgrade-related documents."""
        index_path = self.research_dir / "UPGRADE_INDEX.md"
        if not index_path.exists():
            self.results.append(
                ValidationResult(
                    file="UPGRADE_INDEX.md",
                    check="upgrade_index",
                    passed=False,
                    message="UPGRADE_INDEX.md not found",
                    severity="error",
                )
            )
            return

        index_content = index_path.read_text()

        # Find documents with upgrade references
        for filename, doc in self.documents.items():
            if doc.related_upgrades:
                for upgrade in doc.related_upgrades:
                    if upgrade not in index_content:
                        self.results.append(
                            ValidationResult(
                                file=filename,
                                check="upgrade_index_coverage",
                                passed=False,
                                message=f"{upgrade} not found in UPGRADE_INDEX.md",
                                severity="warning",
                            )
                        )

    def _check_content_quality(self) -> None:
        """Check content quality metrics."""
        for filename, doc in self.documents.items():
            # Skip index and meta files
            if filename in ["README.md", "NAMING_CONVENTION.md", "UPGRADE_INDEX.md"]:
                continue

            # Only check research documents
            if not filename.endswith("_RESEARCH.md"):
                continue

            # Check minimum word count
            if doc.word_count < 500:
                self.results.append(
                    ValidationResult(
                        file=filename,
                        check="content_quality",
                        passed=False,
                        message=f"Document too short ({doc.word_count} words, minimum 500)",
                        severity="warning",
                        suggestion="Add more detailed research content",
                    )
                )

    def _check_duplicate_links(self) -> None:
        """Check for duplicate internal links within documents."""
        for filename, doc in self.documents.items():
            # Skip index files which legitimately have many links
            if filename in ["README.md", "UPGRADE_INDEX.md"]:
                continue

            link_counts: dict[str, int] = {}
            for link in doc.internal_links:
                link_counts[link] = link_counts.get(link, 0) + 1

            for link, count in link_counts.items():
                if count > 3:  # More than 3 is suspicious
                    self.results.append(
                        ValidationResult(
                            file=filename,
                            check="duplicate_links",
                            passed=False,
                            message=f"Link '{link}' appears {count} times",
                            severity="warning",
                            suggestion="Consider consolidating duplicate links",
                        )
                    )

    def get_summary(self) -> dict:
        """Get validation summary."""
        errors = [r for r in self.results if not r.passed and r.severity == "error"]
        warnings = [r for r in self.results if not r.passed and r.severity == "warning"]
        passed = [r for r in self.results if r.passed]

        return {
            "total_checks": len(self.results),
            "passed": len(passed),
            "errors": len(errors),
            "warnings": len(warnings),
            "documents_scanned": len(self.documents),
            "all_passed": len(errors) == 0,
        }

    def auto_fix(self) -> list[str]:
        """Attempt to auto-fix common issues. Returns list of fixed files."""
        fixed_files = []

        for filename, doc in self.documents.items():
            if not doc.has_frontmatter and filename.endswith("_RESEARCH.md"):
                # Add basic frontmatter
                content = doc.path.read_text(encoding="utf-8")
                title = filename.replace("_RESEARCH.md", "").replace("_", " ").title()
                frontmatter = f"""---
title: "{title} Research"
topic: general
related_upgrades: []
related_docs: []
tags: []
created: {datetime.now().strftime("%Y-%m-%d")}
updated: {datetime.now().strftime("%Y-%m-%d")}
---

"""
                new_content = frontmatter + content
                doc.path.write_text(new_content, encoding="utf-8")
                fixed_files.append(filename)
                self.log(f"Added frontmatter to {filename}")

        return fixed_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate research documents against naming conventions")
    parser.add_argument("--fix", action="store_true", help="Attempt to auto-fix issues")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("docs/research"),
        help="Research directory to validate",
    )

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    research_dir = project_root / args.dir

    if not research_dir.exists():
        print(f"Error: Research directory not found: {research_dir}", file=sys.stderr)
        sys.exit(2)

    # Run validation
    validator = ResearchDocValidator(research_dir, verbose=args.verbose)
    results = validator.validate_all()
    summary = validator.get_summary()

    # Auto-fix if requested
    if args.fix:
        print("\n🔧 Attempting auto-fix...")
        fixed = validator.auto_fix()
        if fixed:
            print(f"   Fixed {len(fixed)} file(s): {', '.join(fixed)}")
            # Re-run validation after fixes
            validator = ResearchDocValidator(research_dir, verbose=args.verbose)
            results = validator.validate_all()
            summary = validator.get_summary()
        else:
            print("   No auto-fixable issues found")

    # Output results
    if args.json:
        output = {
            "summary": summary,
            "results": [
                {
                    "file": r.file,
                    "check": r.check,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                    "suggestion": r.suggestion,
                }
                for r in results
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print("\n" + "=" * 60)
        print("Research Document Validation Report")
        print("=" * 60)
        print(f"\nDocuments scanned: {summary['documents_scanned']}")
        print(f"Total checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']}")
        print(f"Errors: {summary['errors']}")
        print(f"Warnings: {summary['warnings']}")

        # Show issues
        issues = [r for r in results if not r.passed]
        if issues:
            print("\n" + "-" * 60)
            print("Issues Found:")
            print("-" * 60)
            for r in issues:
                icon = "❌" if r.severity == "error" else "⚠️"
                print(f"\n{icon} [{r.file}] {r.check}")
                print(f"   {r.message}")
                if r.suggestion:
                    print(f"   💡 Suggestion: {r.suggestion}")

        print("\n" + "=" * 60)
        if summary["all_passed"]:
            print("✅ All validation checks passed!")
        else:
            print(f"❌ Validation failed with {summary['errors']} error(s)")
        print("=" * 60 + "\n")

    # Exit code
    sys.exit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
