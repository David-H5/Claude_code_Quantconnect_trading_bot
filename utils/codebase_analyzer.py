#!/usr/bin/env python3
"""Codebase Analyzer for Redundancy, Duplication, and Conflict Detection.

Scans the trading bot codebase to identify:
- Duplicate functionality across modules
- Redundant implementations
- Potential conflicts between systems
- Overlap in responsibilities

Used by AI agents when planning updates to avoid creating duplicates.

Usage:
    # CLI
    python -m utils.codebase_analyzer --report
    python -m utils.codebase_analyzer --check-conflicts "new feature description"
    python -m utils.codebase_analyzer --find-similar "ClassName or function_name"

    # Python
    from utils.codebase_analyzer import CodebaseAnalyzer, analyze_for_planning

    analyzer = CodebaseAnalyzer()
    report = analyzer.full_analysis()
    conflicts = analyzer.check_planning_conflicts("Add new sentiment analyzer")
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CodeEntity:
    """Represents a code entity (class, function, module)."""

    name: str
    entity_type: str  # class, function, module
    file_path: str
    line_number: int
    docstring: str = ""
    signature: str = ""
    dependencies: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class DuplicationMatch:
    """Represents a potential duplication."""

    entity1: CodeEntity
    entity2: CodeEntity
    similarity_score: float
    match_type: str  # name, functionality, signature
    reason: str


@dataclass
class ConflictWarning:
    """Represents a potential conflict."""

    description: str
    severity: str  # high, medium, low
    affected_files: list[str]
    recommendation: str


@dataclass
class AnalysisReport:
    """Complete analysis report."""

    timestamp: str
    total_files: int
    total_classes: int
    total_functions: int
    duplications: list[DuplicationMatch]
    conflicts: list[ConflictWarning]
    module_overlaps: dict[str, list[str]]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_files": self.total_files,
                "total_classes": self.total_classes,
                "total_functions": self.total_functions,
                "duplication_count": len(self.duplications),
                "conflict_count": len(self.conflicts),
            },
            "duplications": [
                {
                    "entity1": f"{d.entity1.file_path}:{d.entity1.name}",
                    "entity2": f"{d.entity2.file_path}:{d.entity2.name}",
                    "similarity": d.similarity_score,
                    "type": d.match_type,
                    "reason": d.reason,
                }
                for d in self.duplications
            ],
            "conflicts": [
                {
                    "description": c.description,
                    "severity": c.severity,
                    "files": c.affected_files,
                    "recommendation": c.recommendation,
                }
                for c in self.conflicts
            ],
            "module_overlaps": self.module_overlaps,
            "recommendations": self.recommendations,
        }


class CodebaseAnalyzer:
    """Analyzes codebase for redundancy, duplication, and conflicts."""

    # Known functional categories and their keywords
    FUNCTIONAL_CATEGORIES: ClassVar[dict[str, list[str]]] = {
        "sentiment": ["sentiment", "emotion", "tone", "feeling", "mood", "finbert"],
        "anomaly": ["anomaly", "outlier", "abnormal", "spike", "unusual", "flash"],
        "scanner": ["scanner", "scan", "detect", "monitor", "watch"],
        "analyzer": ["analyzer", "analysis", "analyze", "evaluate"],
        "logger": ["logger", "logging", "log", "audit", "trace"],
        "validator": ["validator", "validate", "check", "verify"],
        "monitor": ["monitor", "monitoring", "health", "status", "metric"],
        "executor": ["executor", "execution", "execute", "order", "trade"],
        "strategy": ["strategy", "signal", "entry", "exit", "position"],
        "risk": ["risk", "limit", "circuit", "breaker", "drawdown"],
        "config": ["config", "settings", "parameter", "option"],
        "cache": ["cache", "store", "memory", "buffer"],
    }

    # Known duplicate/redundant patterns to check
    KNOWN_DUPLICATIONS: ClassVar[list[dict[str, Any]]] = [
        {
            "pattern": r"anomaly.*detector",
            "files": ["observability/anomaly_detector.py", "models/anomaly_detector.py"],
            "reason": "Two separate anomaly detectors - agent behavior vs market",
        },
        {
            "pattern": r"sentiment",
            "files": [
                "llm/sentiment.py",
                "llm/emotion_detector.py",
                "llm/reddit_sentiment.py",
                "llm/sentiment_filter.py",
            ],
            "reason": "Multiple sentiment analysis implementations",
        },
        {
            "pattern": r"spread.*analy",
            "files": ["execution/spread_analysis.py", "execution/spread_anomaly.py"],
            "reason": "Spread analysis split across files",
        },
        {
            "pattern": r"logger",
            "files": [
                "llm/reasoning_logger.py",
                "llm/decision_logger.py",
                "observability/exception_logger.py",
                "observability/logging/agent.py",
            ],
            "reason": "Multiple specialized loggers - consider unified logging interface",
        },
        {
            "pattern": r"monitor",
            "files": [
                "models/correlation_monitor.py",
                "models/var_monitor.py",
                "models/greeks_monitor.py",
                "execution/slippage_monitor.py",
                "utils/storage_monitor.py",
                "ui/evolution_monitor.py",
            ],
            "reason": "Monitors spread across modules - use observability/monitoring/",
        },
        {
            "pattern": r"validator",
            "files": [
                "scripts/qa_validator.py",
                "scripts/algorithm_validator.py",
                "execution/pre_trade_validator.py",
                ".claude/hooks/trading/risk_validator.py",
            ],
            "reason": "Validators in multiple locations - consolidate to validation module",
        },
        {
            "pattern": r"monitoring",
            "files": [
                "evaluation/sprint1_monitoring.py",
                "evaluation/continuous_monitoring.py",
                "observability/monitoring/",
            ],
            "reason": "Monitoring code in evaluation/ vs observability/",
        },
    ]

    def __init__(self, project_root: Path | None = None):
        """Initialize analyzer."""
        self.project_root = project_root or PROJECT_ROOT
        self.entities: list[CodeEntity] = []
        self.file_hashes: dict[str, str] = {}
        self._scanned = False

    def scan_codebase(self) -> None:
        """Scan all Python files in the codebase."""
        if self._scanned:
            return

        exclude_dirs = {".venv", "__pycache__", ".git", "node_modules", ".pytest_cache"}

        for py_file in self.project_root.rglob("*.py"):
            # Skip excluded directories
            if any(excl in py_file.parts for excl in exclude_dirs):
                continue

            with contextlib.suppress(Exception):
                self._scan_file(py_file)

        self._scanned = True

    def _scan_file(self, file_path: Path) -> None:
        """Scan a single Python file."""
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        rel_path = str(file_path.relative_to(self.project_root))

        # Calculate file hash for change detection
        self.file_hashes[rel_path] = hashlib.md5(content.encode()).hexdigest()[:12]

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                entity = CodeEntity(
                    name=node.name,
                    entity_type="class",
                    file_path=rel_path,
                    line_number=node.lineno,
                    docstring=ast.get_docstring(node) or "",
                    keywords=self._extract_keywords(node.name, ast.get_docstring(node) or ""),
                )
                self.entities.append(entity)

            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                # Skip private/dunder methods
                if node.name.startswith("_"):
                    continue

                entity = CodeEntity(
                    name=node.name,
                    entity_type="function",
                    file_path=rel_path,
                    line_number=node.lineno,
                    docstring=ast.get_docstring(node) or "",
                    signature=self._get_function_signature(node),
                    keywords=self._extract_keywords(node.name, ast.get_docstring(node) or ""),
                )
                self.entities.append(entity)

    def _extract_keywords(self, name: str, docstring: str) -> list[str]:
        """Extract functional keywords from name and docstring."""
        text = f"{name} {docstring}".lower()
        keywords = []

        for category, terms in self.FUNCTIONAL_CATEGORIES.items():
            if any(term in text for term in terms):
                keywords.append(category)

        return keywords

    def _get_function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Get function signature."""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        return f"({', '.join(args)})"

    def find_duplications(self) -> list[DuplicationMatch]:
        """Find potential duplications in the codebase."""
        self.scan_codebase()
        duplications = []

        # Group entities by keywords
        by_keyword: dict[str, list[CodeEntity]] = defaultdict(list)
        for entity in self.entities:
            for keyword in entity.keywords:
                by_keyword[keyword].append(entity)

        # Find entities with same keywords in different files
        for keyword, entities in by_keyword.items():
            if len(entities) < 2:
                continue

            # Group by file to find cross-file duplications
            by_file: dict[str, list[CodeEntity]] = defaultdict(list)
            for entity in entities:
                by_file[entity.file_path].append(entity)

            # Compare across files
            files = list(by_file.keys())
            for i, file1 in enumerate(files):
                for file2 in files[i + 1 :]:
                    for e1 in by_file[file1]:
                        for e2 in by_file[file2]:
                            if e1.entity_type == e2.entity_type:
                                similarity = self._calculate_similarity(e1, e2)
                                if similarity > 0.6:
                                    duplications.append(
                                        DuplicationMatch(
                                            entity1=e1,
                                            entity2=e2,
                                            similarity_score=similarity,
                                            match_type="functionality",
                                            reason=f"Both implement '{keyword}' functionality",
                                        )
                                    )

        # Check for name-based duplications
        by_name: dict[str, list[CodeEntity]] = defaultdict(list)
        for entity in self.entities:
            # Normalize name
            normalized = re.sub(r"[0-9]+", "", entity.name.lower())
            by_name[normalized].append(entity)

        for _name, entities in by_name.items():
            if len(entities) < 2:
                continue

            # Check if they're in different files
            unique_files = {e.file_path for e in entities}
            if len(unique_files) > 1:
                for i, e1 in enumerate(entities):
                    for e2 in entities[i + 1 :]:
                        if e1.file_path != e2.file_path:
                            duplications.append(
                                DuplicationMatch(
                                    entity1=e1,
                                    entity2=e2,
                                    similarity_score=0.9,
                                    match_type="name",
                                    reason=f"Similar names: {e1.name} vs {e2.name}",
                                )
                            )

        return duplications

    def _calculate_similarity(self, e1: CodeEntity, e2: CodeEntity) -> float:
        """Calculate similarity between two entities."""
        score = 0.0

        # Name similarity
        name1 = e1.name.lower()
        name2 = e2.name.lower()
        if name1 == name2:
            score += 0.4
        elif name1 in name2 or name2 in name1:
            score += 0.2

        # Keyword overlap
        keywords1 = set(e1.keywords)
        keywords2 = set(e2.keywords)
        if keywords1 and keywords2:
            overlap = len(keywords1 & keywords2) / len(keywords1 | keywords2)
            score += 0.3 * overlap

        # Docstring similarity (simple word overlap)
        if e1.docstring and e2.docstring:
            words1 = set(e1.docstring.lower().split())
            words2 = set(e2.docstring.lower().split())
            if words1 and words2:
                overlap = len(words1 & words2) / len(words1 | words2)
                score += 0.3 * overlap

        return min(score, 1.0)

    def find_conflicts(self) -> list[ConflictWarning]:
        """Find potential conflicts in the codebase."""
        self.scan_codebase()
        conflicts = []

        # Check for known duplication patterns
        for pattern_info in self.KNOWN_DUPLICATIONS:
            matching_files = []
            for file_path in pattern_info["files"]:
                if (self.project_root / file_path).exists():
                    matching_files.append(file_path)

            if len(matching_files) > 1:
                conflicts.append(
                    ConflictWarning(
                        description=pattern_info["reason"],
                        severity="medium",
                        affected_files=matching_files,
                        recommendation="Consider consolidating into single module",
                    )
                )

        # Check for competing implementations
        competing_patterns = [
            (r"Logger$", "Multiple logger implementations"),
            (r"Monitor$", "Multiple monitor implementations"),
            (r"Validator$", "Multiple validator implementations"),
            (r"Detector$", "Multiple detector implementations"),
        ]

        for pattern, description in competing_patterns:
            matches = [e for e in self.entities if re.search(pattern, e.name) and e.entity_type == "class"]

            # Group by module
            by_module: dict[str, list[CodeEntity]] = defaultdict(list)
            for entity in matches:
                module = entity.file_path.split("/")[0] if "/" in entity.file_path else ""
                by_module[module].append(entity)

            for module, entities in by_module.items():
                if len(entities) > 2:
                    conflicts.append(
                        ConflictWarning(
                            description=f"{description} in {module}/",
                            severity="low",
                            affected_files=list({e.file_path for e in entities}),
                            recommendation="Review for consolidation opportunities",
                        )
                    )

        return conflicts

    def get_module_overlaps(self) -> dict[str, list[str]]:
        """Identify overlapping responsibilities between modules."""
        self.scan_codebase()
        overlaps: dict[str, list[str]] = defaultdict(list)

        # Group entities by module
        by_module: dict[str, set[str]] = defaultdict(set)
        for entity in self.entities:
            module = entity.file_path.split("/")[0] if "/" in entity.file_path else "root"
            for keyword in entity.keywords:
                by_module[module].add(keyword)

        # Find overlapping keywords between modules
        modules = list(by_module.keys())
        for i, mod1 in enumerate(modules):
            for mod2 in modules[i + 1 :]:
                shared = by_module[mod1] & by_module[mod2]
                if shared:
                    key = f"{mod1} <-> {mod2}"
                    overlaps[key] = list(shared)

        return dict(overlaps)

    def check_planning_conflicts(self, proposed_feature: str) -> dict[str, Any]:
        """Check if a proposed feature conflicts with existing code.

        Args:
            proposed_feature: Description of the proposed feature/change

        Returns:
            Dictionary with conflict analysis
        """
        self.scan_codebase()

        # Extract keywords from proposal
        proposal_lower = proposed_feature.lower()
        matched_categories = []

        for category, terms in self.FUNCTIONAL_CATEGORIES.items():
            if any(term in proposal_lower for term in terms):
                matched_categories.append(category)

        # Find existing implementations
        existing_implementations = []
        for entity in self.entities:
            if any(cat in entity.keywords for cat in matched_categories):
                existing_implementations.append(
                    {
                        "name": entity.name,
                        "type": entity.entity_type,
                        "file": entity.file_path,
                        "line": entity.line_number,
                        "categories": entity.keywords,
                    }
                )

        # Generate warnings
        warnings = []
        if existing_implementations:
            warnings.append(
                f"Found {len(existing_implementations)} existing implementations "
                f"related to: {', '.join(matched_categories)}"
            )

            # Group by file for consolidation suggestions
            by_file: dict[str, list[dict]] = defaultdict(list)
            for impl in existing_implementations:
                by_file[impl["file"]].append(impl)

            if len(by_file) > 1:
                warnings.append(
                    f"Implementations spread across {len(by_file)} files - "
                    "consider consolidating before adding new functionality"
                )

        # Generate recommendations
        recommendations = []
        if existing_implementations:
            recommendations.append("Review existing implementations before creating new ones:")
            for impl in existing_implementations[:5]:  # Show top 5
                recommendations.append(f"  - {impl['file']}:{impl['line']} - {impl['name']}")

            if len(existing_implementations) > 5:
                recommendations.append(f"  ... and {len(existing_implementations) - 5} more")

            recommendations.append("")
            recommendations.append("Consider:")
            recommendations.append("  1. Extending existing classes instead of creating new ones")
            recommendations.append("  2. Adding to existing modules rather than new files")
            recommendations.append("  3. Using composition with existing implementations")

        return {
            "proposed_feature": proposed_feature,
            "matched_categories": matched_categories,
            "existing_count": len(existing_implementations),
            "existing_implementations": existing_implementations[:10],
            "warnings": warnings,
            "recommendations": recommendations,
            "conflict_risk": "high"
            if len(existing_implementations) > 3
            else "medium"
            if existing_implementations
            else "low",
        }

    def full_analysis(self) -> AnalysisReport:
        """Run complete codebase analysis."""
        self.scan_codebase()

        duplications = self.find_duplications()
        conflicts = self.find_conflicts()
        overlaps = self.get_module_overlaps()

        # Generate recommendations
        recommendations = []

        if duplications:
            recommendations.append(f"Found {len(duplications)} potential duplications - review for consolidation")

        if conflicts:
            high_severity = [c for c in conflicts if c.severity == "high"]
            if high_severity:
                recommendations.append(f"Address {len(high_severity)} high-severity conflicts before major changes")

        # Module-specific recommendations
        sentiment_files = [e.file_path for e in self.entities if "sentiment" in e.keywords]
        if len(set(sentiment_files)) > 3:
            recommendations.append("Sentiment analysis is fragmented across 5+ files - consolidate to single module")

        anomaly_files = [e.file_path for e in self.entities if "anomaly" in e.keywords]
        if len(set(anomaly_files)) > 1:
            recommendations.append("Anomaly detection split between observability/ and models/ - consider unifying")

        classes = [e for e in self.entities if e.entity_type == "class"]
        functions = [e for e in self.entities if e.entity_type == "function"]

        return AnalysisReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_files=len(self.file_hashes),
            total_classes=len(classes),
            total_functions=len(functions),
            duplications=duplications,
            conflicts=conflicts,
            module_overlaps=overlaps,
            recommendations=recommendations,
        )

    def generate_planning_context(self) -> str:
        """Generate context string for AI agent planning."""
        report = self.full_analysis()

        lines = [
            "# Codebase Analysis for Planning",
            "",
            f"**Scanned**: {report.total_files} files, {report.total_classes} classes, {report.total_functions} functions",
            "",
            "## Known Duplications/Redundancies",
            "",
        ]

        # Add known duplications
        for pattern_info in self.KNOWN_DUPLICATIONS:
            lines.append(f"- **{pattern_info['reason']}**")
            for f in pattern_info["files"]:
                lines.append(f"  - `{f}`")
            lines.append("")

        lines.extend(
            [
                "## Module Overlaps",
                "",
            ]
        )

        for overlap, keywords in report.module_overlaps.items():
            if keywords:
                lines.append(f"- {overlap}: {', '.join(keywords)}")

        lines.extend(
            [
                "",
                "## Recommendations",
                "",
            ]
        )

        for rec in report.recommendations:
            lines.append(f"- {rec}")

        lines.extend(
            [
                "",
                "## Before Creating New Code",
                "",
                "1. Search for existing implementations using `check_planning_conflicts()`",
                "2. Check if functionality can be added to existing modules",
                "3. Prefer extending existing classes over creating new ones",
                "4. Update docs/ARCHITECTURE.md if adding new modules",
                "",
            ]
        )

        return "\n".join(lines)


def analyze_for_planning(feature_description: str) -> dict[str, Any]:
    """Quick analysis for planning a new feature.

    Args:
        feature_description: Description of proposed feature

    Returns:
        Conflict analysis with recommendations
    """
    analyzer = CodebaseAnalyzer()
    return analyzer.check_planning_conflicts(feature_description)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Codebase analyzer for redundancy and conflict detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate full analysis report",
    )
    parser.add_argument(
        "--check-conflicts",
        type=str,
        metavar="DESCRIPTION",
        help="Check if proposed feature conflicts with existing code",
    )
    parser.add_argument(
        "--find-similar",
        type=str,
        metavar="NAME",
        help="Find entities with similar names or functionality",
    )
    parser.add_argument(
        "--planning-context",
        action="store_true",
        help="Generate planning context for AI agents",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )

    args = parser.parse_args()

    analyzer = CodebaseAnalyzer()

    if args.check_conflicts:
        result = analyzer.check_planning_conflicts(args.check_conflicts)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n=== Conflict Analysis: {args.check_conflicts} ===\n")
            print(f"Risk Level: {result['conflict_risk'].upper()}")
            print(f"Related Categories: {', '.join(result['matched_categories']) or 'None'}")
            print(f"Existing Implementations: {result['existing_count']}")

            if result["warnings"]:
                print("\nWarnings:")
                for warning in result["warnings"]:
                    print(f"  ⚠️  {warning}")

            if result["recommendations"]:
                print("\nRecommendations:")
                for rec in result["recommendations"]:
                    print(f"  {rec}")

        return 0

    if args.find_similar:
        analyzer.scan_codebase()
        search_term = args.find_similar.lower()

        matches = []
        for entity in analyzer.entities:
            if search_term in entity.name.lower() or search_term in entity.docstring.lower():
                matches.append(entity)

        if args.json:
            print(
                json.dumps(
                    [
                        {"name": m.name, "type": m.entity_type, "file": m.file_path, "line": m.line_number}
                        for m in matches
                    ],
                    indent=2,
                )
            )
        else:
            print(f"\n=== Similar Entities: {args.find_similar} ===\n")
            if matches:
                for m in matches:
                    print(f"  {m.entity_type}: {m.name}")
                    print(f"    File: {m.file_path}:{m.line_number}")
                    if m.docstring:
                        first_line = m.docstring.split("\n")[0][:80]
                        print(f"    Doc: {first_line}")
                    print()
            else:
                print("  No matches found")

        return 0

    if args.planning_context:
        context = analyzer.generate_planning_context()
        print(context)
        return 0

    # Default: full report
    report = analyzer.full_analysis()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("\n=== Codebase Analysis Report ===\n")
        print(f"Timestamp: {report.timestamp}")
        print(f"Files: {report.total_files}")
        print(f"Classes: {report.total_classes}")
        print(f"Functions: {report.total_functions}")

        print(f"\n--- Duplications ({len(report.duplications)}) ---")
        for dup in report.duplications[:10]:
            print(f"  • {dup.entity1.name} ({dup.entity1.file_path})")
            print(f"    ↔ {dup.entity2.name} ({dup.entity2.file_path})")
            print(f"    Reason: {dup.reason}")
            print()

        print(f"\n--- Conflicts ({len(report.conflicts)}) ---")
        for conflict in report.conflicts:
            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(conflict.severity, "⚪")
            print(f"  {severity_icon} {conflict.description}")
            print(f"     Files: {', '.join(conflict.affected_files)}")
            print(f"     Recommendation: {conflict.recommendation}")
            print()

        print("\n--- Recommendations ---")
        for rec in report.recommendations:
            print(f"  • {rec}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
