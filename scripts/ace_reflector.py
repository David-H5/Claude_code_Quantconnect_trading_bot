#!/usr/bin/env python3
"""ACE Reflector Module - Pattern Extraction from Session Outcomes.

Implements the Reflector component from Stanford's ACE (Agentic Context Engineering)
framework (arXiv:2510.04618). Analyzes session outcomes to extract patterns,
identify misclassifications, and generate recommendations for improving task routing.

Reference: https://arxiv.org/abs/2510.04618

Usage:
    python scripts/ace_reflector.py --analyze
    python scripts/ace_reflector.py --analyze --days 7
    python scripts/ace_reflector.py --output recommendations.json
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class PatternRecommendation:
    """A recommendation for improving task routing."""

    pattern_type: str  # keyword_gap, misclassification, complexity_mismatch, domain_miss
    confidence: float
    description: str
    action: str  # add_l3_pattern, add_l2_pattern, add_depth_indicator, add_width_indicator
    pattern: str
    weight: int
    justification: str
    supporting_sessions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pattern_type": self.pattern_type,
            "confidence": self.confidence,
            "description": self.description,
            "recommendation": {
                "action": self.action,
                "pattern": self.pattern,
                "weight": self.weight,
                "justification": self.justification,
            },
            "supporting_evidence": [{"session_id": sid} for sid in self.supporting_sessions[:5]],
        }


@dataclass
class ReflectorAnalysis:
    """Complete analysis output from the Reflector."""

    analysis_date: str
    sessions_analyzed: int
    date_range: dict
    patterns_found: list[PatternRecommendation]
    overall_metrics: dict
    top_keywords_in_failures: list[tuple[str, int]]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "analysis_date": self.analysis_date,
            "sessions_analyzed": self.sessions_analyzed,
            "date_range": self.date_range,
            "patterns_found": [p.to_dict() for p in self.patterns_found],
            "overall_metrics": self.overall_metrics,
            "top_keywords_in_failures": [{"keyword": kw, "count": cnt} for kw, cnt in self.top_keywords_in_failures],
        }


class ACEReflector:
    """Implements the Reflector component of the ACE framework.

    Analyzes session outcomes to extract patterns and generate recommendations
    for improving task classification accuracy.

    Key responsibilities (per ACE framework):
    - Analyze outcomes from task execution
    - Extract insights from successes and failures
    - Identify missing heuristics or rules
    - Diagnose failure modes
    """

    # Common words to exclude from keyword extraction
    STOP_WORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "not",
        "only",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "then",
        "here",
        "there",
        "up",
        "down",
    }

    def __init__(self, log_path: Path | None = None):
        """Initialize the Reflector.

        Args:
            log_path: Path to session-outcomes.jsonl file.
        """
        self.log_path = log_path or Path("logs/session-outcomes.jsonl")

    def load_sessions(self, days: int | None = None, since: datetime | None = None) -> list[dict]:
        """Load session outcomes from JSONL file.

        Args:
            days: Only include sessions from the last N days.
            since: Only include sessions since this datetime.

        Returns:
            List of session outcome dictionaries.
        """
        if not self.log_path.exists():
            return []

        sessions = []
        cutoff = None

        if days:
            cutoff = datetime.now() - timedelta(days=days)
        elif since:
            cutoff = since

        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    session = json.loads(line)
                    if cutoff:
                        ts = session.get("timestamp", "")
                        if ts:
                            try:
                                session_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                if session_time.replace(tzinfo=None) < cutoff:
                                    continue
                            except ValueError:
                                pass
                    sessions.append(session)
                except json.JSONDecodeError:
                    continue

        return sessions

    def extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text.

        Args:
            text: The text to extract keywords from.

        Returns:
            List of keywords (lowercased, filtered).
        """
        # Tokenize and clean
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text.lower())

        # Filter stop words and short words
        keywords = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]

        return keywords

    def analyze(self, days: int | None = None) -> ReflectorAnalysis:
        """Analyze session outcomes and extract patterns.

        This is the core Reflector function that:
        1. Loads session outcomes
        2. Identifies misclassifications and failures
        3. Extracts keyword patterns from failures
        4. Generates recommendations for improving routing

        Args:
            days: Only analyze sessions from the last N days.

        Returns:
            ReflectorAnalysis with patterns and recommendations.
        """
        sessions = self.load_sessions(days=days)

        if not sessions:
            return ReflectorAnalysis(
                analysis_date=datetime.now().isoformat(),
                sessions_analyzed=0,
                date_range={"start": None, "end": None},
                patterns_found=[],
                overall_metrics={},
                top_keywords_in_failures=[],
            )

        # Calculate date range
        timestamps = [s.get("timestamp", "") for s in sessions if s.get("timestamp")]
        date_range = {
            "start": min(timestamps) if timestamps else None,
            "end": max(timestamps) if timestamps else None,
        }

        # Categorize sessions
        misclassified = []  # classification_accurate = false
        failed = []  # status = failed
        partial = []  # status = partial
        successful = []  # status = success

        # Track by complexity level
        by_complexity = defaultdict(list)
        long_running_simple = []  # L1 tasks > 60 min

        for session in sessions:
            routing = session.get("routing_decision", {})
            outcome = session.get("outcome", {})
            feedback = session.get("feedback", {})

            complexity = routing.get("complexity_level", "unknown")
            status = outcome.get("status", "unknown")
            duration = outcome.get("duration_minutes", 0)

            by_complexity[complexity].append(session)

            # Check for misclassification feedback
            if feedback.get("classification_accurate") is False:
                misclassified.append(session)

            # Categorize by outcome
            if status == "failed":
                failed.append(session)
            elif status == "partial":
                partial.append(session)
            elif status == "success":
                successful.append(session)

            # Check for long-running simple tasks
            if complexity == "L1_simple" and duration > 60:
                long_running_simple.append(session)

        # Extract patterns and generate recommendations
        patterns = []

        # Pattern 1: Keywords in misclassified tasks
        if misclassified:
            patterns.extend(self._analyze_misclassified(misclassified))

        # Pattern 2: Keywords in failed tasks
        if failed:
            patterns.extend(self._analyze_failures(failed, "failed"))

        # Pattern 3: Long-running simple tasks (should be L2/L3)
        if long_running_simple:
            patterns.extend(self._analyze_long_running(long_running_simple))

        # Pattern 4: Complexity level performance
        patterns.extend(self._analyze_complexity_performance(by_complexity))

        # Calculate overall metrics
        total = len(sessions)
        overall_metrics = {
            "total_sessions": total,
            "success_rate": len(successful) / total if total > 0 else 0,
            "failure_rate": len(failed) / total if total > 0 else 0,
            "partial_rate": len(partial) / total if total > 0 else 0,
            "misclassification_rate": len(misclassified) / total if total > 0 else 0,
            "success_by_level": {
                level: sum(1 for s in sessions_list if s.get("outcome", {}).get("status") == "success")
                / len(sessions_list)
                if sessions_list
                else 0
                for level, sessions_list in by_complexity.items()
            },
        }

        # Extract top keywords from failures for manual review
        all_failure_keywords = []
        for session in failed + partial + misclassified:
            task = session.get("task_description", "")
            all_failure_keywords.extend(self.extract_keywords(task))

        keyword_counts = Counter(all_failure_keywords)
        top_keywords = keyword_counts.most_common(20)

        return ReflectorAnalysis(
            analysis_date=datetime.now().isoformat(),
            sessions_analyzed=total,
            date_range=date_range,
            patterns_found=patterns,
            overall_metrics=overall_metrics,
            top_keywords_in_failures=top_keywords,
        )

    def _analyze_misclassified(self, sessions: list[dict]) -> list[PatternRecommendation]:
        """Analyze misclassified tasks to find keyword gaps.

        Args:
            sessions: Sessions marked as misclassified.

        Returns:
            List of pattern recommendations.
        """
        patterns = []

        # Extract keywords from misclassified tasks
        keyword_sessions = defaultdict(list)
        for session in sessions:
            task = session.get("task_description", "")
            session_id = session.get("session_id", "unknown")
            keywords = self.extract_keywords(task)
            for kw in keywords:
                keyword_sessions[kw].append(session_id)

        # Find keywords that appear in multiple misclassified sessions
        for keyword, session_ids in keyword_sessions.items():
            if len(session_ids) >= 2:
                patterns.append(
                    PatternRecommendation(
                        pattern_type="keyword_gap",
                        confidence=min(0.5 + len(session_ids) * 0.1, 0.95),
                        description=f"Tasks containing '{keyword}' frequently misclassified",
                        action="add_l3_pattern",
                        pattern=keyword,
                        weight=2,
                        justification=f"Found in {len(session_ids)} misclassified sessions",
                        supporting_sessions=session_ids[:5],
                    )
                )

        return patterns

    def _analyze_failures(self, sessions: list[dict], status: str) -> list[PatternRecommendation]:
        """Analyze failed tasks to find patterns.

        Args:
            sessions: Sessions that failed.
            status: The status label (failed/partial).

        Returns:
            List of pattern recommendations.
        """
        patterns = []

        # Group by complexity level
        by_level = defaultdict(list)
        for session in sessions:
            level = session.get("routing_decision", {}).get("complexity_level", "unknown")
            by_level[level].append(session)

        # Check if simple tasks are failing at high rate
        simple_failures = by_level.get("L1_simple", [])
        if len(simple_failures) >= 3:
            # Extract common keywords
            keyword_counts = Counter()
            for session in simple_failures:
                task = session.get("task_description", "")
                keyword_counts.update(self.extract_keywords(task))

            top_keyword = keyword_counts.most_common(1)
            if top_keyword:
                kw, count = top_keyword[0]
                patterns.append(
                    PatternRecommendation(
                        pattern_type="complexity_mismatch",
                        confidence=min(0.6 + count * 0.1, 0.9),
                        description=f"L1 tasks with '{kw}' frequently {status}",
                        action="add_l2_pattern",
                        pattern=kw,
                        weight=2,
                        justification=f"Keyword '{kw}' found in {count} {status} L1 tasks",
                        supporting_sessions=[s.get("session_id", "unknown") for s in simple_failures[:5]],
                    )
                )

        return patterns

    def _analyze_long_running(self, sessions: list[dict]) -> list[PatternRecommendation]:
        """Analyze long-running simple tasks.

        Args:
            sessions: L1 tasks that took > 60 minutes.

        Returns:
            List of pattern recommendations.
        """
        patterns = []

        if len(sessions) >= 2:
            # Extract common keywords
            keyword_counts = Counter()
            for session in sessions:
                task = session.get("task_description", "")
                keyword_counts.update(self.extract_keywords(task))

            top_keywords = keyword_counts.most_common(3)
            for kw, count in top_keywords:
                if count >= 2:
                    patterns.append(
                        PatternRecommendation(
                            pattern_type="complexity_mismatch",
                            confidence=min(0.6 + count * 0.1, 0.85),
                            description=f"L1 tasks with '{kw}' consistently take > 60 minutes",
                            action="add_depth_indicator",
                            pattern=kw,
                            weight=2,
                            justification=f"'{kw}' appears in {count} long-running L1 tasks",
                            supporting_sessions=[s.get("session_id", "unknown") for s in sessions[:5]],
                        )
                    )

        return patterns

    def _analyze_complexity_performance(self, by_complexity: dict[str, list]) -> list[PatternRecommendation]:
        """Analyze performance by complexity level.

        Args:
            by_complexity: Sessions grouped by complexity level.

        Returns:
            List of pattern recommendations.
        """
        patterns = []

        for level, sessions in by_complexity.items():
            if not sessions:
                continue

            success_count = sum(1 for s in sessions if s.get("outcome", {}).get("status") == "success")
            success_rate = success_count / len(sessions)

            # Flag levels with very low success rate
            if success_rate < 0.5 and len(sessions) >= 5:
                patterns.append(
                    PatternRecommendation(
                        pattern_type="domain_miss",
                        confidence=0.7,
                        description=f"{level} tasks have only {success_rate:.0%} success rate",
                        action="review_level_thresholds",
                        pattern=level,
                        weight=0,
                        justification=f"{success_count}/{len(sessions)} sessions succeeded",
                        supporting_sessions=[
                            s.get("session_id", "unknown")
                            for s in sessions
                            if s.get("outcome", {}).get("status") != "success"
                        ][:5],
                    )
                )

        return patterns

    def generate_report(self, analysis: ReflectorAnalysis) -> str:
        """Generate a human-readable report from analysis.

        Args:
            analysis: The analysis results.

        Returns:
            Formatted report string.
        """
        lines = [
            "# ACE Reflector Analysis Report",
            "",
            f"**Analysis Date**: {analysis.analysis_date}",
            f"**Sessions Analyzed**: {analysis.sessions_analyzed}",
            f"**Date Range**: {analysis.date_range.get('start', 'N/A')} to " f"{analysis.date_range.get('end', 'N/A')}",
            "",
            "## Overall Metrics",
            "",
        ]

        metrics = analysis.overall_metrics
        if metrics:
            lines.extend(
                [
                    f"- **Success Rate**: {metrics.get('success_rate', 0):.1%}",
                    f"- **Failure Rate**: {metrics.get('failure_rate', 0):.1%}",
                    f"- **Partial Rate**: {metrics.get('partial_rate', 0):.1%}",
                    f"- **Misclassification Rate**: " f"{metrics.get('misclassification_rate', 0):.1%}",
                    "",
                    "### Success by Complexity Level",
                    "",
                ]
            )
            for level, rate in metrics.get("success_by_level", {}).items():
                lines.append(f"- **{level}**: {rate:.1%}")

        lines.extend(
            [
                "",
                "## Patterns Found",
                "",
            ]
        )

        if analysis.patterns_found:
            for i, pattern in enumerate(analysis.patterns_found, 1):
                lines.extend(
                    [
                        f"### Pattern {i}: {pattern.pattern_type}",
                        "",
                        f"**Description**: {pattern.description}",
                        f"**Confidence**: {pattern.confidence:.0%}",
                        "",
                        "**Recommendation**:",
                        f"- Action: `{pattern.action}`",
                        f"- Pattern: `{pattern.pattern}`",
                        f"- Weight: {pattern.weight}",
                        f"- Justification: {pattern.justification}",
                        "",
                    ]
                )
        else:
            lines.append("No significant patterns found.")

        lines.extend(
            [
                "",
                "## Top Keywords in Failures/Misclassifications",
                "",
            ]
        )

        if analysis.top_keywords_in_failures:
            for kw, count in analysis.top_keywords_in_failures[:10]:
                lines.append(f"- `{kw}`: {count} occurrences")
        else:
            lines.append("No failure keywords extracted.")

        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ACE Reflector - Analyze session outcomes for patterns")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis on session outcomes",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Only analyze sessions from last N days",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for recommendations",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate human-readable report",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        help="Path to session-outcomes.jsonl (default: logs/session-outcomes.jsonl)",
    )

    args = parser.parse_args()

    if not args.analyze and not args.report:
        args.analyze = True  # Default to analyze

    reflector = ACEReflector(log_path=args.log_path)
    analysis = reflector.analyze(days=args.days)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(analysis.to_dict(), f, indent=2)
        print(f"Analysis saved to {args.output}", file=sys.stderr)

    if args.report or (not args.output):
        report = reflector.generate_report(analysis)
        print(report)

    # Return success if we found patterns or analyzed sessions
    return 0 if analysis.sessions_analyzed > 0 or args.output else 1


if __name__ == "__main__":
    sys.exit(main())
