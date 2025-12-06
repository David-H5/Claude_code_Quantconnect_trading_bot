#!/usr/bin/env python3
"""
RIC Loop Convergence Scorer

Calculates convergence score for Enhanced RIC Loop using the formula:
Score = (0.4 x success_criteria) + (0.3 x P0_complete) + (0.2 x test_coverage) + (0.1 x P1_complete)

Usage:
    python scripts/ric_convergence.py                           # Interactive mode
    python scripts/ric_convergence.py --check                   # Check current score
    python scripts/ric_convergence.py --score 0.85 0.90 0.75 0.60  # Direct calculation
    python scripts/ric_convergence.py --record 0.82             # Record score to session
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / ".claude" / "hooks"))

try:
    from ric_state_manager import Priority, RICStateManager

    HAS_RIC_STATE = True
except ImportError:
    HAS_RIC_STATE = False
    RICStateManager = None
    Priority = None


# Score weights (from CLAUDE.md)
WEIGHTS = {
    "success_criteria": 0.4,
    "p0_complete": 0.3,
    "test_coverage": 0.2,
    "p1_complete": 0.1,
}

# Exit thresholds
SUCCESS_THRESHOLD = 0.80
PLATEAU_THRESHOLD = 0.02
MAX_ITERATIONS = 5


def calculate_score(
    success_criteria: float,
    p0_complete: float,
    test_coverage: float,
    p1_complete: float,
) -> float:
    """
    Calculate convergence score.

    Args:
        success_criteria: Percentage of success criteria met (0.0 - 1.0)
        p0_complete: Percentage of P0 items complete (0.0 - 1.0)
        test_coverage: Test coverage percentage (0.0 - 1.0)
        p1_complete: Percentage of P1 items complete (0.0 - 1.0)

    Returns:
        Composite score between 0.0 and 1.0
    """
    score = (
        WEIGHTS["success_criteria"] * success_criteria
        + WEIGHTS["p0_complete"] * p0_complete
        + WEIGHTS["test_coverage"] * test_coverage
        + WEIGHTS["p1_complete"] * p1_complete
    )
    return round(score, 4)


def format_score_breakdown(
    success_criteria: float,
    p0_complete: float,
    test_coverage: float,
    p1_complete: float,
) -> str:
    """Format score breakdown for display."""
    lines = [
        "### Composite Score Calculation",
        f"- Success criteria: {success_criteria:.0%} x 0.4 = {success_criteria * 0.4:.4f}",
        f"- P0 complete: {p0_complete:.0%} x 0.3 = {p0_complete * 0.3:.4f}",
        f"- Test coverage: {test_coverage:.0%} x 0.2 = {test_coverage * 0.2:.4f}",
        f"- P1 complete: {p1_complete:.0%} x 0.1 = {p1_complete * 0.1:.4f}",
    ]

    total = calculate_score(success_criteria, p0_complete, test_coverage, p1_complete)
    lines.append(f"- **TOTAL SCORE**: {total:.4f}")

    return "\n".join(lines)


def check_exit_condition(
    score: float,
    previous_scores: list | None = None,
    iteration: int = 1,
) -> tuple[bool, str]:
    """
    Check if loop should exit.

    Returns:
        (should_exit, reason)
    """
    # Success threshold
    if score >= SUCCESS_THRESHOLD:
        return True, "SUCCESS"

    # Max iterations
    if iteration >= MAX_ITERATIONS:
        return True, "MAX_ITERATIONS"

    # Plateau detection
    if previous_scores and len(previous_scores) >= 2:
        recent_scores = previous_scores[-2:] + [score]
        deltas = [abs(recent_scores[i + 1] - recent_scores[i]) for i in range(len(recent_scores) - 1)]

        if all(d < PLATEAU_THRESHOLD for d in deltas):
            return True, "PLATEAU"

    return False, "CONTINUE"


def get_recommendation(score: float, exit_reason: str) -> str:
    """Get recommendation based on score and exit reason."""
    if exit_reason == "SUCCESS":
        return (
            "Score >= 0.80: Ready to exit!\n"
            "- Complete final documentation\n"
            "- Create checkpoint commit\n"
            "- Update project status"
        )
    elif exit_reason == "PLATEAU":
        return (
            "Plateau detected: Consider exiting.\n"
            "- Score improvement < 2% for 2 iterations\n"
            "- Document what was achieved\n"
            "- Create backlog for remaining items"
        )
    elif exit_reason == "MAX_ITERATIONS":
        return (
            "Maximum iterations reached: Must exit.\n"
            "- Document current state\n"
            "- Note why convergence wasn't achieved\n"
            "- Create comprehensive backlog\n"
            "- Flag for human review"
        )
    else:
        if score >= 0.70:
            return (
                "Good progress! Continue to next iteration.\n"
                "- Focus on remaining P0 items\n"
                "- Consider if more research needed"
            )
        elif score >= 0.50:
            return (
                "Moderate progress. Continue iterating.\n"
                "- Review success criteria\n"
                "- May need additional research\n"
                "- Consider breaking down complex items"
            )
        else:
            return (
                "Low score - significant work remains.\n"
                "- Review upgrade path\n"
                "- Consider simpler approach\n"
                "- May need to reduce scope"
            )


def get_session_info() -> dict[str, Any] | None:
    """Get session info from the new state manager."""
    if not HAS_RIC_STATE or RICStateManager is None:
        return None

    manager = RICStateManager()
    if not manager.is_active():
        return None

    state = manager.get_state()
    open_insights = manager.get_open_insights()

    # Calculate insight completion percentages
    total_insights = len(state.insights)
    resolved_insights = total_insights - len(open_insights)

    p0_total = len([i for i in state.insights if i.priority == Priority.P0])
    p0_open = len([i for i in open_insights if i.priority == Priority.P0])
    p1_total = len([i for i in state.insights if i.priority == Priority.P1])
    p1_open = len([i for i in open_insights if i.priority == Priority.P1])

    return {
        "session_id": state.upgrade_id,
        "task": state.title,
        "current_iteration": state.current_iteration,
        "max_iterations": state.max_iterations,
        "current_phase": state.current_phase.value,
        "p0_total": p0_total,
        "p0_completed": p0_total - p0_open,
        "p1_total": p1_total,
        "p1_completed": p1_total - p1_open,
        "total_insights": total_insights,
        "resolved_insights": resolved_insights,
    }


def interactive_mode() -> None:
    """Run interactive convergence calculation."""
    print("=" * 60)
    print("RIC Loop Convergence Calculator")
    print("=" * 60)

    # Get current session info if available
    if HAS_RIC_STATE:
        summary = get_session_info()
        if summary:
            print(f"\nActive session: {summary['session_id']}")
            print(f"Task: {summary['task']}")
            print(f"Iteration: {summary['current_iteration']}/{summary['max_iterations']}")
            print(f"P0: {summary['p0_completed']}/{summary['p0_total']} complete")
            print(f"P1: {summary['p1_completed']}/{summary['p1_total']} complete")
        else:
            print("\nNo active RIC session found.")
    print()

    # Get inputs
    try:
        success_criteria = float(input("Success criteria met (0-100%): ").strip().rstrip("%")) / 100
        p0_complete = float(input("P0 items complete (0-100%): ").strip().rstrip("%")) / 100
        test_coverage = float(input("Test coverage (0-100%): ").strip().rstrip("%")) / 100
        p1_complete = float(input("P1 items complete (0-100%): ").strip().rstrip("%")) / 100
    except ValueError as e:
        print(f"Invalid input: {e}")
        sys.exit(1)

    # Calculate
    score = calculate_score(success_criteria, p0_complete, test_coverage, p1_complete)

    print("\n" + "-" * 60)
    print(format_score_breakdown(success_criteria, p0_complete, test_coverage, p1_complete))

    # Check exit conditions
    previous_scores = []
    iteration = 1
    if HAS_RIC_STATE:
        summary = get_session_info()
        if summary:
            iteration = summary.get("current_iteration", 1)

    should_exit_flag, reason = check_exit_condition(score, previous_scores, iteration)

    print("\n" + "-" * 60)
    print("### Exit Condition Check")
    print(f"- Score: {score:.4f}")
    print(f"- Threshold: {SUCCESS_THRESHOLD}")
    print(f"- Decision: **{reason}**")

    # Also check state manager exit eligibility
    if HAS_RIC_STATE and RICStateManager is not None:
        manager = RICStateManager()
        if manager.is_active():
            can_exit, exit_reason = manager.can_exit()
            print(f"- RIC Exit Status: {'ALLOWED' if can_exit else 'BLOCKED'}")
            print(f"- RIC Exit Reason: {exit_reason}")

    print("\n" + "-" * 60)
    print("### Recommendation")
    print(get_recommendation(score, reason))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RIC Loop Convergence Scorer")
    parser.add_argument("--check", action="store_true", help="Check current session score")
    parser.add_argument(
        "--score",
        nargs=4,
        type=float,
        metavar=("SUCCESS", "P0", "COVERAGE", "P1"),
        help="Calculate score from values (0-1 scale)",
    )
    parser.add_argument("--record", type=float, help="Record score to current session")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    args = parser.parse_args()

    if args.check:
        if not HAS_RIC_STATE or RICStateManager is None:
            print("RIC state module not available")
            sys.exit(1)

        manager = RICStateManager()
        summary = get_session_info()
        if summary:
            can_exit, exit_reason = manager.can_exit()
            if args.json:
                summary["can_exit"] = can_exit
                summary["exit_reason"] = exit_reason
                print(json.dumps(summary, indent=2))
            else:
                print(f"Session: {summary['session_id']}")
                print(f"Task: {summary['task']}")
                print(f"Iteration: {summary['current_iteration']}/{summary['max_iterations']}")
                print(f"Phase: {summary['current_phase']}")
                print(f"P0 Complete: {summary['p0_completed']}/{summary['p0_total']}")
                print(f"P1 Complete: {summary['p1_completed']}/{summary['p1_total']}")
                print(f"Exit Status: {'ALLOWED' if can_exit else 'BLOCKED'} ({exit_reason})")
        else:
            print("No active session")
        return

    if args.score:
        score = calculate_score(*args.score)
        should_exit_flag, reason = check_exit_condition(score)

        if args.json:
            print(
                json.dumps(
                    {
                        "score": score,
                        "should_exit": should_exit_flag,
                        "reason": reason,
                        "breakdown": {
                            "success_criteria": args.score[0] * 0.4,
                            "p0_complete": args.score[1] * 0.3,
                            "test_coverage": args.score[2] * 0.2,
                            "p1_complete": args.score[3] * 0.1,
                        },
                    },
                    indent=2,
                )
            )
        else:
            print(format_score_breakdown(*args.score))
            print(f"\nDecision: {reason}")
        return

    if args.record is not None:
        # Note: The new RICStateManager tracks progress via P0/P1/P2 insights
        # rather than convergence scores. Use add-insight command instead.
        print("Note: Score recording is deprecated in the new RIC state manager.")
        print(f"Requested score: {args.record}")
        print("\nProgress is now tracked via insights. Use:")
        print("  python3 .claude/hooks/ric_state_manager.py add-insight --priority P0 --description '...'")
        print("  python3 .claude/hooks/ric_state_manager.py status")
        return

    # Default: interactive mode
    interactive_mode()


if __name__ == "__main__":
    main()
