#!/usr/bin/env python3
"""
Session Outcome Analyzer for Hierarchical Prompt System (UPGRADE-012.1).

Analyzes session outcomes to identify patterns, trends, and areas for improvement.

Usage:
    python scripts/analyze_session_outcomes.py
    python scripts/analyze_session_outcomes.py --report
    python scripts/analyze_session_outcomes.py --weekly
    python scripts/analyze_session_outcomes.py --export-csv outcomes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class AnalysisResult:
    """Result of session outcome analysis."""

    total_sessions: int
    success_rate: float
    partial_rate: float
    failed_rate: float
    avg_duration_mins: float
    avg_tasks_completed: float
    avg_task_completion_rate: float
    by_complexity: dict
    by_domain: dict
    classification_accuracy: float | None
    trends: dict


class SessionOutcomeAnalyzer:
    """Analyzer for session outcomes."""

    def __init__(self, log_file: Path | None = None):
        """Initialize analyzer.

        Args:
            log_file: Path to session outcomes JSONL file.
        """
        if log_file is None:
            project_root = Path(__file__).parent.parent
            log_file = project_root / "logs" / "session-outcomes.jsonl"

        self.log_file = log_file
        self.outcomes = self._load_outcomes()

    def _load_outcomes(self) -> list[dict]:
        """Load outcomes from JSONL file."""
        if not self.log_file.exists():
            return []

        outcomes = []
        with open(self.log_file) as f:
            for line in f:
                if line.strip():
                    outcomes.append(json.loads(line))

        return outcomes

    def analyze(self, days: int | None = None) -> AnalysisResult:
        """Analyze session outcomes.

        Args:
            days: Only analyze outcomes from the last N days. None for all.

        Returns:
            AnalysisResult with comprehensive analysis.
        """
        outcomes = self.outcomes

        # Filter by date if specified
        if days is not None:
            cutoff = datetime.now() - timedelta(days=days)
            outcomes = [o for o in outcomes if datetime.fromisoformat(o.get("timestamp", "2000-01-01")) >= cutoff]

        if not outcomes:
            return AnalysisResult(
                total_sessions=0,
                success_rate=0.0,
                partial_rate=0.0,
                failed_rate=0.0,
                avg_duration_mins=0.0,
                avg_tasks_completed=0.0,
                avg_task_completion_rate=0.0,
                by_complexity={},
                by_domain={},
                classification_accuracy=None,
                trends={},
            )

        # Basic stats
        total = len(outcomes)
        success_count = sum(1 for o in outcomes if o.get("outcome", {}).get("status") == "success")
        partial_count = sum(1 for o in outcomes if o.get("outcome", {}).get("status") == "partial")
        failed_count = sum(1 for o in outcomes if o.get("outcome", {}).get("status") == "failed")

        # Duration and tasks
        durations = [o.get("outcome", {}).get("duration_minutes") for o in outcomes]
        durations = [d for d in durations if d is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        tasks_completed = [o.get("outcome", {}).get("tasks_completed", 0) for o in outcomes]
        avg_tasks = sum(tasks_completed) / len(tasks_completed) if tasks_completed else 0.0

        completion_rates = []
        for o in outcomes:
            completed = o.get("outcome", {}).get("tasks_completed", 0)
            total_tasks = o.get("outcome", {}).get("tasks_total", 0)
            if total_tasks > 0:
                completion_rates.append(completed / total_tasks)

        avg_completion_rate = sum(completion_rates) / len(completion_rates) if completion_rates else 0.0

        # By complexity
        by_complexity = self._analyze_by_field(outcomes, "routing_decision", "complexity_level")

        # By domain
        by_domain = self._analyze_by_field(outcomes, "routing_decision", "domain")

        # Classification accuracy
        accuracy = self._calculate_classification_accuracy(outcomes)

        # Trends
        trends = self._calculate_trends(outcomes)

        return AnalysisResult(
            total_sessions=total,
            success_rate=success_count / total if total > 0 else 0.0,
            partial_rate=partial_count / total if total > 0 else 0.0,
            failed_rate=failed_count / total if total > 0 else 0.0,
            avg_duration_mins=avg_duration,
            avg_tasks_completed=avg_tasks,
            avg_task_completion_rate=avg_completion_rate,
            by_complexity=by_complexity,
            by_domain=by_domain,
            classification_accuracy=accuracy,
            trends=trends,
        )

    def _analyze_by_field(self, outcomes: list[dict], parent_key: str, field: str) -> dict:
        """Analyze outcomes grouped by a specific field."""
        grouped: dict[str, list] = defaultdict(list)

        for o in outcomes:
            value = o.get(parent_key, {}).get(field, "unknown")
            grouped[value].append(o)

        result = {}
        for key, group in grouped.items():
            total = len(group)
            success = sum(1 for o in group if o.get("outcome", {}).get("status") == "success")
            result[key] = {
                "count": total,
                "success_rate": success / total if total > 0 else 0.0,
                "avg_score": sum(o.get("routing_decision", {}).get("complexity_score", 0) for o in group) / total
                if total > 0
                else 0.0,
            }

        return result

    def _calculate_classification_accuracy(self, outcomes: list[dict]) -> float | None:
        """Calculate classification accuracy from feedback."""
        with_feedback = [o for o in outcomes if o.get("feedback", {}).get("classification_accurate") is not None]

        if not with_feedback:
            return None

        accurate = sum(1 for o in with_feedback if o.get("feedback", {}).get("classification_accurate") is True)

        return accurate / len(with_feedback)

    def _calculate_trends(self, outcomes: list[dict]) -> dict:
        """Calculate trends over time."""
        if len(outcomes) < 2:
            return {"trend_direction": "insufficient_data"}

        # Sort by timestamp
        sorted_outcomes = sorted(outcomes, key=lambda o: o.get("timestamp", "2000-01-01"))

        # Split into halves
        mid = len(sorted_outcomes) // 2
        first_half = sorted_outcomes[:mid]
        second_half = sorted_outcomes[mid:]

        # Calculate success rates
        first_success = (
            sum(1 for o in first_half if o.get("outcome", {}).get("status") == "success") / len(first_half)
            if first_half
            else 0
        )

        second_success = (
            sum(1 for o in second_half if o.get("outcome", {}).get("status") == "success") / len(second_half)
            if second_half
            else 0
        )

        # Determine trend
        diff = second_success - first_success
        if diff > 0.1:
            direction = "improving"
        elif diff < -0.1:
            direction = "declining"
        else:
            direction = "stable"

        return {
            "trend_direction": direction,
            "first_half_success_rate": first_success,
            "second_half_success_rate": second_success,
            "change": diff,
        }

    def generate_report(self, result: AnalysisResult) -> str:
        """Generate human-readable report.

        Args:
            result: Analysis result to report on.

        Returns:
            Formatted report string.
        """
        if result.total_sessions == 0:
            return "No session outcomes to analyze.\n"

        report = []
        report.append("=" * 60)
        report.append("SESSION OUTCOME ANALYSIS REPORT (UPGRADE-012.1)")
        report.append("=" * 60)
        report.append("")

        # Overview
        report.append("## Overview")
        report.append(f"Total Sessions: {result.total_sessions}")
        report.append(f"Success Rate: {result.success_rate:.1%}")
        report.append(f"Partial Rate: {result.partial_rate:.1%}")
        report.append(f"Failed Rate: {result.failed_rate:.1%}")
        report.append("")

        # Performance
        report.append("## Performance Metrics")
        report.append(f"Avg Duration: {result.avg_duration_mins:.1f} minutes")
        report.append(f"Avg Tasks Completed: {result.avg_tasks_completed:.1f}")
        report.append(f"Avg Completion Rate: {result.avg_task_completion_rate:.1%}")
        report.append("")

        # By Complexity
        if result.by_complexity:
            report.append("## By Complexity Level")
            for level, stats in sorted(result.by_complexity.items()):
                report.append(f"  {level}:")
                report.append(f"    Sessions: {stats['count']}")
                report.append(f"    Success Rate: {stats['success_rate']:.1%}")
                report.append(f"    Avg Score: {stats['avg_score']:.1f}")
            report.append("")

        # By Domain
        if result.by_domain:
            report.append("## By Domain")
            for domain, stats in sorted(result.by_domain.items()):
                report.append(f"  {domain}:")
                report.append(f"    Sessions: {stats['count']}")
                report.append(f"    Success Rate: {stats['success_rate']:.1%}")
            report.append("")

        # Classification Accuracy
        if result.classification_accuracy is not None:
            report.append("## Classification Accuracy")
            report.append(f"Accuracy: {result.classification_accuracy:.1%}")
            report.append("(Based on user feedback)")
            report.append("")

        # Trends
        if result.trends.get("trend_direction") != "insufficient_data":
            report.append("## Trends")
            report.append(f"Direction: {result.trends['trend_direction'].upper()}")
            report.append(f"First Half Success: {result.trends['first_half_success_rate']:.1%}")
            report.append(f"Second Half Success: {result.trends['second_half_success_rate']:.1%}")
            report.append(f"Change: {result.trends['change']:+.1%}")
            report.append("")

        # Recommendations
        report.append("## Recommendations")
        if result.success_rate < 0.5:
            report.append("- Low success rate: Review classification patterns")
        if result.avg_task_completion_rate < 0.7:
            report.append("- Task completion rate low: Consider breaking tasks smaller")
        if result.classification_accuracy is not None and result.classification_accuracy < 0.8:
            report.append("- Classification accuracy low: Adjust keyword patterns")
        if result.trends.get("trend_direction") == "declining":
            report.append("- Performance declining: Investigate recent changes")
        if not report[-1].startswith("-"):
            report.append("- All metrics look healthy!")
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)

    def export_csv(self, output_path: Path) -> int:
        """Export outcomes to CSV file.

        Args:
            output_path: Path to output CSV file.

        Returns:
            Number of rows exported.
        """
        if not self.outcomes:
            return 0

        fieldnames = [
            "session_id",
            "timestamp",
            "task_description",
            "complexity_level",
            "complexity_score",
            "depth_score",
            "width_score",
            "domain",
            "status",
            "tasks_completed",
            "tasks_total",
            "duration_minutes",
            "classification_accurate",
            "workflow_helpful",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for o in self.outcomes:
                row = {
                    "session_id": o.get("session_id"),
                    "timestamp": o.get("timestamp"),
                    "task_description": o.get("task_description", "")[:100],
                    "complexity_level": o.get("routing_decision", {}).get("complexity_level"),
                    "complexity_score": o.get("routing_decision", {}).get("complexity_score"),
                    "depth_score": o.get("routing_decision", {}).get("depth_score"),
                    "width_score": o.get("routing_decision", {}).get("width_score"),
                    "domain": o.get("routing_decision", {}).get("domain"),
                    "status": o.get("outcome", {}).get("status"),
                    "tasks_completed": o.get("outcome", {}).get("tasks_completed"),
                    "tasks_total": o.get("outcome", {}).get("tasks_total"),
                    "duration_minutes": o.get("outcome", {}).get("duration_minutes"),
                    "classification_accurate": o.get("feedback", {}).get("classification_accurate"),
                    "workflow_helpful": o.get("feedback", {}).get("workflow_helpful"),
                }
                writer.writerow(row)

        return len(self.outcomes)

    def get_session_notes_summary(self) -> str:
        """Generate summary suitable for session notes.

        Returns:
            Markdown summary for session notes.
        """
        result = self.analyze(days=7)  # Last 7 days

        if result.total_sessions == 0:
            return "No recent session outcomes to summarize.\n"

        summary = []
        summary.append("### Session Outcome Trends (Last 7 Days)")
        summary.append("")
        summary.append(f"- **Sessions**: {result.total_sessions}")
        summary.append(f"- **Success Rate**: {result.success_rate:.1%}")
        summary.append(f"- **Avg Completion**: {result.avg_task_completion_rate:.1%}")

        if result.trends.get("trend_direction") != "insufficient_data":
            direction = result.trends["trend_direction"]
            emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(direction, "")
            summary.append(f"- **Trend**: {emoji} {direction}")

        summary.append("")

        return "\n".join(summary)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze session outcomes for patterns and trends.")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed analysis report",
    )
    parser.add_argument(
        "--weekly",
        action="store_true",
        help="Analyze only the last 7 days",
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Analyze only the last N days",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        metavar="FILE",
        help="Export outcomes to CSV file",
    )
    parser.add_argument(
        "--session-notes",
        action="store_true",
        help="Generate summary for session notes",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output analysis as JSON",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to session-outcomes.jsonl file",
    )

    args = parser.parse_args()

    analyzer = SessionOutcomeAnalyzer(log_file=args.log_file)

    # Determine analysis period
    days = args.days
    if args.weekly:
        days = 7

    # Export CSV
    if args.export_csv:
        count = analyzer.export_csv(args.export_csv)
        print(f"Exported {count} outcomes to {args.export_csv}")
        return

    # Session notes summary
    if args.session_notes:
        print(analyzer.get_session_notes_summary())
        return

    # Analysis
    result = analyzer.analyze(days=days)

    if args.json:
        output = {
            "total_sessions": result.total_sessions,
            "success_rate": result.success_rate,
            "partial_rate": result.partial_rate,
            "failed_rate": result.failed_rate,
            "avg_duration_mins": result.avg_duration_mins,
            "avg_tasks_completed": result.avg_tasks_completed,
            "avg_task_completion_rate": result.avg_task_completion_rate,
            "by_complexity": result.by_complexity,
            "by_domain": result.by_domain,
            "classification_accuracy": result.classification_accuracy,
            "trends": result.trends,
        }
        print(json.dumps(output, indent=2))
    elif args.report:
        print(analyzer.generate_report(result))
    else:
        # Quick summary
        print("Session Outcome Summary")
        print("-" * 30)
        print(f"Total Sessions: {result.total_sessions}")
        print(f"Success Rate: {result.success_rate:.1%}")
        print(f"Avg Completion: {result.avg_task_completion_rate:.1%}")
        if result.trends.get("trend_direction") != "insufficient_data":
            print(f"Trend: {result.trends['trend_direction']}")
        print()
        print("Use --report for detailed analysis")


if __name__ == "__main__":
    main()
