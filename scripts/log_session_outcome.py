#!/usr/bin/env python3
"""
Session Outcome Logger for Hierarchical Prompt System (UPGRADE-012.1).

Logs session outcomes to enable future analysis and learning.
This implements the ACE (Agentic Context Engineering) feedback loop.

Usage:
    python scripts/log_session_outcome.py --session-id "20251203-021500" \
        --status success --tasks-completed 5 --tasks-total 6 \
        --task "Implement feature X" --complexity L1_moderate --domain algorithm

    python scripts/log_session_outcome.py --interactive

Schema:
    See docs/research/UPGRADE-012.1-DEPTH-WIDTH-CLASSIFICATION.md for full schema.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class RoutingDecision:
    """Routing decision from task router."""

    complexity_level: str
    complexity_score: int
    depth_score: float
    width_score: float
    domain: str


@dataclass
class SessionOutcome:
    """Outcome of a Claude Code session."""

    status: str  # success, partial, failed
    tasks_completed: int
    tasks_total: int
    errors_encountered: list[str] = field(default_factory=list)
    duration_minutes: float | None = None


@dataclass
class SessionFeedback:
    """Optional human feedback on session."""

    classification_accurate: bool | None = None
    workflow_helpful: bool | None = None
    notes: str | None = None


@dataclass
class SessionOutcomeLog:
    """Complete session outcome log entry."""

    session_id: str
    task_description: str
    routing_decision: RoutingDecision
    outcome: SessionOutcome
    feedback: SessionFeedback
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "task_description": self.task_description,
            "routing_decision": {
                "complexity_level": self.routing_decision.complexity_level,
                "complexity_score": self.routing_decision.complexity_score,
                "depth_score": self.routing_decision.depth_score,
                "width_score": self.routing_decision.width_score,
                "domain": self.routing_decision.domain,
            },
            "outcome": {
                "status": self.outcome.status,
                "tasks_completed": self.outcome.tasks_completed,
                "tasks_total": self.outcome.tasks_total,
                "errors_encountered": self.outcome.errors_encountered,
                "duration_minutes": self.outcome.duration_minutes,
            },
            "feedback": {
                "classification_accurate": self.feedback.classification_accurate,
                "workflow_helpful": self.feedback.workflow_helpful,
                "notes": self.feedback.notes,
            },
            "timestamp": self.timestamp,
        }


class SessionOutcomeLogger:
    """Logger for session outcomes."""

    def __init__(self, log_file: Path | None = None):
        """Initialize logger.

        Args:
            log_file: Path to JSONL log file. Defaults to logs/session-outcomes.jsonl.
        """
        if log_file is None:
            project_root = Path(__file__).parent.parent
            log_file = project_root / "logs" / "session-outcomes.jsonl"

        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: SessionOutcomeLog) -> None:
        """Log a session outcome entry.

        Args:
            entry: The session outcome to log.
        """
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def get_recent_outcomes(self, limit: int = 10) -> list[dict]:
        """Get recent session outcomes.

        Args:
            limit: Maximum number of outcomes to return.

        Returns:
            List of outcome dictionaries, most recent first.
        """
        if not self.log_file.exists():
            return []

        outcomes = []
        with open(self.log_file) as f:
            for line in f:
                if line.strip():
                    outcomes.append(json.loads(line))

        return outcomes[-limit:][::-1]

    def get_classification_accuracy(self) -> dict:
        """Calculate classification accuracy from feedback.

        Returns:
            Dictionary with accuracy metrics.
        """
        if not self.log_file.exists():
            return {"total": 0, "with_feedback": 0, "accurate": 0, "accuracy_rate": None}

        total = 0
        with_feedback = 0
        accurate = 0

        with open(self.log_file) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    total += 1
                    feedback = entry.get("feedback", {})
                    if feedback.get("classification_accurate") is not None:
                        with_feedback += 1
                        if feedback["classification_accurate"]:
                            accurate += 1

        return {
            "total": total,
            "with_feedback": with_feedback,
            "accurate": accurate,
            "accuracy_rate": accurate / with_feedback if with_feedback > 0 else None,
        }

    def get_success_rate_by_complexity(self) -> dict:
        """Calculate success rate by complexity level.

        Returns:
            Dictionary with success rates per complexity level.
        """
        if not self.log_file.exists():
            return {}

        stats: dict[str, dict] = {}

        with open(self.log_file) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    level = entry.get("routing_decision", {}).get("complexity_level", "unknown")
                    status = entry.get("outcome", {}).get("status", "unknown")

                    if level not in stats:
                        stats[level] = {"total": 0, "success": 0, "partial": 0, "failed": 0}

                    stats[level]["total"] += 1
                    if status in ("success", "partial", "failed"):
                        stats[level][status] += 1

        # Calculate rates
        for level, data in stats.items():
            total = data["total"]
            if total > 0:
                data["success_rate"] = data["success"] / total
                data["completion_rate"] = (data["success"] + data["partial"]) / total

        return stats


def create_outcome_from_args(args: argparse.Namespace) -> SessionOutcomeLog:
    """Create SessionOutcomeLog from command line arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        SessionOutcomeLog instance.
    """
    routing = RoutingDecision(
        complexity_level=args.complexity,
        complexity_score=args.complexity_score or 0,
        depth_score=args.depth_score or 0.0,
        width_score=args.width_score or 0.0,
        domain=args.domain,
    )

    outcome = SessionOutcome(
        status=args.status,
        tasks_completed=args.tasks_completed,
        tasks_total=args.tasks_total,
        errors_encountered=args.errors or [],
        duration_minutes=args.duration,
    )

    feedback = SessionFeedback(
        classification_accurate=args.classification_accurate,
        workflow_helpful=args.workflow_helpful,
        notes=args.notes,
    )

    return SessionOutcomeLog(
        session_id=args.session_id,
        task_description=args.task,
        routing_decision=routing,
        outcome=outcome,
        feedback=feedback,
    )


def run_interactive() -> SessionOutcomeLog:
    """Run interactive session to collect outcome data.

    Returns:
        SessionOutcomeLog instance.
    """
    print("=== Session Outcome Logger (Interactive) ===\n")

    # Session ID
    default_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = input(f"Session ID [{default_id}]: ").strip() or default_id

    # Task description
    task = input("Task description: ").strip()
    if not task:
        task = "Unknown task"

    # Routing decision
    print("\n--- Routing Decision ---")
    complexity = input("Complexity level [L1_simple/L1_moderate/L1_complex]: ").strip()
    if complexity not in ("L1_simple", "L1_moderate", "L1_complex"):
        complexity = "L1_moderate"

    complexity_score = input("Complexity score [0]: ").strip()
    complexity_score = int(complexity_score) if complexity_score else 0

    depth_score = input("Depth score [0.0]: ").strip()
    depth_score = float(depth_score) if depth_score else 0.0

    width_score = input("Width score [0.0]: ").strip()
    width_score = float(width_score) if width_score else 0.0

    domain = input("Domain [general]: ").strip() or "general"

    routing = RoutingDecision(
        complexity_level=complexity,
        complexity_score=complexity_score,
        depth_score=depth_score,
        width_score=width_score,
        domain=domain,
    )

    # Outcome
    print("\n--- Session Outcome ---")
    status = input("Status [success/partial/failed]: ").strip()
    if status not in ("success", "partial", "failed"):
        status = "partial"

    tasks_completed = input("Tasks completed [0]: ").strip()
    tasks_completed = int(tasks_completed) if tasks_completed else 0

    tasks_total = input("Tasks total [0]: ").strip()
    tasks_total = int(tasks_total) if tasks_total else 0

    errors_str = input("Errors (comma-separated) []: ").strip()
    errors = [e.strip() for e in errors_str.split(",") if e.strip()] if errors_str else []

    duration_str = input("Duration in minutes []: ").strip()
    duration = float(duration_str) if duration_str else None

    outcome = SessionOutcome(
        status=status,
        tasks_completed=tasks_completed,
        tasks_total=tasks_total,
        errors_encountered=errors,
        duration_minutes=duration,
    )

    # Feedback
    print("\n--- Feedback (optional) ---")
    classification_str = input("Classification accurate? [y/n/skip]: ").strip().lower()
    classification_accurate = None
    if classification_str == "y":
        classification_accurate = True
    elif classification_str == "n":
        classification_accurate = False

    workflow_str = input("Workflow helpful? [y/n/skip]: ").strip().lower()
    workflow_helpful = None
    if workflow_str == "y":
        workflow_helpful = True
    elif workflow_str == "n":
        workflow_helpful = False

    notes = input("Notes []: ").strip() or None

    feedback = SessionFeedback(
        classification_accurate=classification_accurate,
        workflow_helpful=workflow_helpful,
        notes=notes,
    )

    return SessionOutcomeLog(
        session_id=session_id,
        task_description=task,
        routing_decision=routing,
        outcome=outcome,
        feedback=feedback,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Log session outcomes for analysis and learning.")

    # Mode selection
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode to collect outcome data",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics from logged outcomes",
    )
    parser.add_argument(
        "--recent",
        type=int,
        metavar="N",
        help="Show N most recent outcomes",
    )

    # Session info
    parser.add_argument("--session-id", help="Session identifier")
    parser.add_argument("--task", help="Task description")

    # Routing decision
    parser.add_argument(
        "--complexity",
        choices=["L1_simple", "L1_moderate", "L1_complex"],
        help="Complexity level",
    )
    parser.add_argument("--complexity-score", type=int, help="Complexity score")
    parser.add_argument("--depth-score", type=float, help="Depth score")
    parser.add_argument("--width-score", type=float, help="Width score")
    parser.add_argument("--domain", help="Domain")

    # Outcome
    parser.add_argument(
        "--status",
        choices=["success", "partial", "failed"],
        help="Session status",
    )
    parser.add_argument("--tasks-completed", type=int, help="Number of tasks completed")
    parser.add_argument("--tasks-total", type=int, help="Total number of tasks")
    parser.add_argument("--errors", nargs="*", help="Errors encountered")
    parser.add_argument("--duration", type=float, help="Duration in minutes")

    # Feedback
    parser.add_argument(
        "--classification-accurate",
        type=lambda x: x.lower() == "true",
        help="Was classification accurate? (true/false)",
    )
    parser.add_argument(
        "--workflow-helpful",
        type=lambda x: x.lower() == "true",
        help="Was workflow helpful? (true/false)",
    )
    parser.add_argument("--notes", help="Additional notes")

    # Output
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file",
    )

    args = parser.parse_args()

    logger = SessionOutcomeLogger(log_file=args.log_file)

    # Handle stats mode
    if args.stats:
        print("=== Session Outcome Statistics ===\n")

        accuracy = logger.get_classification_accuracy()
        print("Classification Accuracy:")
        print(f"  Total sessions: {accuracy['total']}")
        print(f"  With feedback: {accuracy['with_feedback']}")
        if accuracy["accuracy_rate"] is not None:
            print(f"  Accuracy rate: {accuracy['accuracy_rate']:.1%}")
        else:
            print("  Accuracy rate: N/A (no feedback)")

        print("\nSuccess Rate by Complexity:")
        stats = logger.get_success_rate_by_complexity()
        for level, data in sorted(stats.items()):
            print(f"  {level}:")
            print(f"    Total: {data['total']}")
            if "success_rate" in data:
                print(f"    Success rate: {data['success_rate']:.1%}")
                print(f"    Completion rate: {data['completion_rate']:.1%}")

        return

    # Handle recent mode
    if args.recent is not None:
        outcomes = logger.get_recent_outcomes(limit=args.recent)
        if not outcomes:
            print("No outcomes logged yet.")
            return

        print(f"=== Recent {len(outcomes)} Sessions ===\n")
        for outcome in outcomes:
            print(f"Session: {outcome['session_id']}")
            print(f"  Task: {outcome['task_description'][:60]}...")
            print(f"  Complexity: {outcome['routing_decision']['complexity_level']}")
            print(f"  Status: {outcome['outcome']['status']}")
            print(f"  Tasks: {outcome['outcome']['tasks_completed']}/{outcome['outcome']['tasks_total']}")
            print()

        return

    # Handle interactive mode
    if args.interactive:
        entry = run_interactive()
        logger.log(entry)
        print(f"\nLogged session outcome to {logger.log_file}")
        print(json.dumps(entry.to_dict(), indent=2))
        return

    # Handle command-line logging
    if not all([args.session_id, args.task, args.complexity, args.domain, args.status]):
        parser.error(
            "When not using --interactive, --stats, or --recent, "
            "you must provide: --session-id, --task, --complexity, --domain, --status"
        )

    if args.tasks_completed is None or args.tasks_total is None:
        parser.error("You must provide --tasks-completed and --tasks-total")

    entry = create_outcome_from_args(args)
    logger.log(entry)
    print(f"Logged session outcome to {logger.log_file}")
    print(json.dumps(entry.to_dict(), indent=2))


if __name__ == "__main__":
    main()
