#!/usr/bin/env python3
"""Weekly Agent Evaluation Pipeline.

Runs weekly (or on-demand) to:
1. Evaluate all agents over past week
2. Identify top/bottom performers
3. Extract weakness categories
4. Generate prompt improvements
5. Store results for A/B testing

Usage:
    python scripts/weekly_agent_evaluation.py [--days N] [--json]
"""

import contextlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class AgentEvaluation:
    """Evaluation results for a single agent."""

    agent_name: str
    decision_count: int
    accuracy: float  # Correct predictions / total
    avg_confidence: float
    confidence_calibration: float  # How well confidence matches accuracy
    weaknesses: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)
    action_distribution: dict[str, int] = field(default_factory=dict)
    daily_activity: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_name": self.agent_name,
            "decision_count": self.decision_count,
            "accuracy": self.accuracy,
            "avg_confidence": self.avg_confidence,
            "confidence_calibration": self.confidence_calibration,
            "weaknesses": self.weaknesses,
            "strengths": self.strengths,
            "improvement_suggestions": self.improvement_suggestions,
            "action_distribution": self.action_distribution,
            "daily_activity": self.daily_activity,
        }


# Confidence level to numeric mapping
CONFIDENCE_MAP = {
    "very_low": 0.1,
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7,
    "very_high": 0.9,
}

# Weakness detection rules
WEAKNESS_RULES = [
    {
        "condition": lambda e: e.avg_confidence < 0.4,
        "weakness": "Low confidence in decisions",
        "suggestion": "Add more data sources to increase signal strength",
    },
    {
        "condition": lambda e: e.avg_confidence > 0.85 and e.accuracy < 0.6,
        "weakness": "Overconfident - high confidence but low accuracy",
        "suggestion": "Add calibration training to align confidence with outcomes",
    },
    {
        "condition": lambda e: len(e.action_distribution) == 1,
        "weakness": "Single action type - lacks decision diversity",
        "suggestion": "Review decision thresholds - may be too conservative or aggressive",
    },
    {
        "condition": lambda e: e.decision_count < 5,
        "weakness": "Low activity - insufficient data for evaluation",
        "suggestion": "Increase opportunity detection sensitivity",
    },
    {
        "condition": lambda e: abs(e.confidence_calibration) > 0.2,
        "weakness": "Poor calibration - confidence doesn't match outcomes",
        "suggestion": "Implement confidence recalibration based on historical accuracy",
    },
]

# Strength detection rules
STRENGTH_RULES = [
    {
        "condition": lambda e: e.avg_confidence > 0.7 and e.accuracy > 0.7,
        "strength": "Well-calibrated high confidence",
    },
    {
        "condition": lambda e: len(e.action_distribution) >= 3,
        "strength": "Diverse decision-making across multiple action types",
    },
    {
        "condition": lambda e: e.decision_count >= 20,
        "strength": "Active agent with substantial decision history",
    },
    {
        "condition": lambda e: abs(e.confidence_calibration) < 0.1,
        "strength": "Excellent confidence calibration",
    },
]


def load_decisions(decision_dir: Path, since: datetime) -> dict[str, list[dict[str, Any]]]:
    """Load decisions from log files grouped by agent.

    Args:
        decision_dir: Directory containing decision log files.
        since: Only load decisions after this datetime.

    Returns:
        Dictionary mapping agent names to their decision lists.
    """
    agent_decisions: dict[str, list[dict[str, Any]]] = {}

    if not decision_dir.exists():
        return agent_decisions

    for log_file in decision_dir.glob("*.json"):
        try:
            data = json.loads(log_file.read_text())
            timestamp_str = data.get("timestamp", "")

            # Parse timestamp
            with contextlib.suppress(ValueError):
                decision_time = datetime.fromisoformat(timestamp_str)
                if decision_time < since:
                    continue

            agent = data.get("agent_name", "unknown")
            if agent not in agent_decisions:
                agent_decisions[agent] = []
            agent_decisions[agent].append(data)
        except (json.JSONDecodeError, OSError):
            continue

    return agent_decisions


def evaluate_agent(agent_name: str, decisions: list[dict[str, Any]]) -> AgentEvaluation:
    """Evaluate a single agent's performance.

    Args:
        agent_name: Name of the agent being evaluated.
        decisions: List of decision records for this agent.

    Returns:
        AgentEvaluation with computed metrics and insights.
    """
    if not decisions:
        return AgentEvaluation(
            agent_name=agent_name,
            decision_count=0,
            accuracy=0.0,
            avg_confidence=0.0,
            confidence_calibration=0.0,
            weaknesses=["No decisions to evaluate"],
            improvement_suggestions=["Ensure agent is receiving opportunities"],
        )

    # Calculate confidence metrics
    confidences = []
    for d in decisions:
        conf_level = d.get("confidence")
        if isinstance(conf_level, float):
            confidences.append(conf_level)
        else:
            conf_str = d.get("overall_confidence", "medium")
            confidences.append(CONFIDENCE_MAP.get(conf_str, 0.5))

    avg_conf = sum(confidences) / len(confidences)

    # Track action distribution
    action_counts: dict[str, int] = {}
    for d in decisions:
        action = d.get("action") or d.get("decision", "").split()[0].lower()
        action = action if action else "unknown"
        action_counts[action] = action_counts.get(action, 0) + 1

    # Track daily activity
    daily_counts: dict[str, int] = {}
    for d in decisions:
        timestamp_str = d.get("timestamp", "")
        with contextlib.suppress(ValueError):
            dt = datetime.fromisoformat(timestamp_str)
            day_key = dt.strftime("%Y-%m-%d")
            daily_counts[day_key] = daily_counts.get(day_key, 0) + 1

    # Calculate accuracy from outcomes (if available)
    correct = 0
    total_with_outcome = 0
    for d in decisions:
        outcome = d.get("outcome")
        if outcome in ["executed", "correct", "successful"]:
            correct += 1
            total_with_outcome += 1
        elif outcome in ["rejected", "incorrect", "failed"]:
            total_with_outcome += 1

    accuracy = correct / total_with_outcome if total_with_outcome > 0 else 0.5

    # Calculate calibration (difference between confidence and accuracy)
    calibration = accuracy - avg_conf

    # Create evaluation object
    evaluation = AgentEvaluation(
        agent_name=agent_name,
        decision_count=len(decisions),
        accuracy=accuracy,
        avg_confidence=avg_conf,
        confidence_calibration=calibration,
        action_distribution=action_counts,
        daily_activity=daily_counts,
    )

    # Apply weakness rules
    for rule in WEAKNESS_RULES:
        if rule["condition"](evaluation):
            evaluation.weaknesses.append(rule["weakness"])
            if "suggestion" in rule:
                evaluation.improvement_suggestions.append(rule["suggestion"])

    # Apply strength rules
    for rule in STRENGTH_RULES:
        if rule["condition"](evaluation):
            evaluation.strengths.append(rule["strength"])

    return evaluation


def run_weekly_evaluation(days: int = 7) -> dict[str, AgentEvaluation]:
    """Run evaluation for all agents.

    Args:
        days: Number of days to look back for decisions.

    Returns:
        Dictionary mapping agent names to their evaluations.
    """
    decision_dir = Path(".claude/state/decisions")
    since = datetime.now() - timedelta(days=days)

    # Load all decisions
    agent_decisions = load_decisions(decision_dir, since)

    # Evaluate each agent
    evaluations = {}
    for agent, decisions in agent_decisions.items():
        evaluations[agent] = evaluate_agent(agent, decisions)

    # Save results
    output_dir = Path(".claude/state/evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"weekly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        "evaluation_date": datetime.now().isoformat(),
        "period_days": days,
        "agent_count": len(evaluations),
        "agents": {agent: e.to_dict() for agent, e in evaluations.items()},
    }
    output_file.write_text(json.dumps(output_data, indent=2))

    return evaluations


def print_summary(evaluations: dict[str, AgentEvaluation]) -> None:
    """Print human-readable evaluation summary.

    Args:
        evaluations: Dictionary of agent evaluations.
    """
    print("\n" + "=" * 70)
    print("WEEKLY AGENT EVALUATION REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    if not evaluations:
        print("\nNo agents to evaluate.")
        return

    # Sort by decision count (activity)
    sorted_agents = sorted(evaluations.values(), key=lambda e: e.decision_count, reverse=True)

    print(f"\n{'Agent':<25} {'Decisions':<12} {'Confidence':<12} {'Status':<15}")
    print("-" * 70)

    for eval_result in sorted_agents:
        status = "⚠️ Weak" if eval_result.weaknesses else "✅ Good"
        if eval_result.strengths and not eval_result.weaknesses:
            status = "🌟 Strong"

        print(
            f"{eval_result.agent_name:<25} "
            f"{eval_result.decision_count:<12} "
            f"{eval_result.avg_confidence:<12.2f} "
            f"{status:<15}"
        )

    # Detail sections
    for eval_result in sorted_agents:
        if eval_result.weaknesses or eval_result.strengths:
            print(f"\n--- {eval_result.agent_name} ---")

            if eval_result.strengths:
                print("  Strengths:")
                for s in eval_result.strengths:
                    print(f"    ✓ {s}")

            if eval_result.weaknesses:
                print("  Weaknesses:")
                for w in eval_result.weaknesses:
                    print(f"    ✗ {w}")

            if eval_result.improvement_suggestions:
                print("  Suggestions:")
                for s in eval_result.improvement_suggestions:
                    print(f"    → {s}")

    print("\n" + "=" * 70)


def main() -> int:
    """Main entry point for weekly evaluation.

    Returns:
        Exit code (0 for success).
    """
    # Parse arguments
    days = 7
    json_output = "--json" in sys.argv

    if "--days" in sys.argv:
        idx = sys.argv.index("--days")
        if idx + 1 < len(sys.argv):
            with contextlib.suppress(ValueError):
                days = int(sys.argv[idx + 1])

    print(f"Running agent evaluation for past {days} days...")

    evaluations = run_weekly_evaluation(days)

    if json_output:
        output = {agent: e.to_dict() for agent, e in evaluations.items()}
        print(json.dumps(output, indent=2))
    else:
        print_summary(evaluations)

    return 0


if __name__ == "__main__":
    sys.exit(main())
