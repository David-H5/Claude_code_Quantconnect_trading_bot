"""Real-time agent performance leaderboard.

Provides ranking and comparison of trading agents based on:
- Decision accuracy (correct predictions)
- Confidence calibration
- Activity level
- Trend analysis (improving/declining)

Usage:
    from observability.agent_leaderboard import get_leaderboard, print_leaderboard

    rankings = get_leaderboard(days=30)
    print_leaderboard()

    # Or run directly
    python -m observability.agent_leaderboard
"""

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class TrendDirection(Enum):
    """Trend direction for agent performance."""

    UP = "up"
    DOWN = "down"
    STABLE = "stable"


@dataclass
class AgentRanking:
    """Performance ranking for a single agent.

    Attributes:
        agent_name: Name of the agent.
        rank: Current ranking position (1 = best).
        score: Composite performance score (0-100).
        accuracy: Accuracy rate if outcomes available.
        avg_confidence: Average confidence level.
        decision_count: Number of decisions in period.
        trend: Performance trend direction.
        rank_change: Change in rank from previous period.
        metrics: Additional metric details.
    """

    agent_name: str
    rank: int
    score: float
    accuracy: float
    avg_confidence: float
    decision_count: int
    trend: TrendDirection = TrendDirection.STABLE
    rank_change: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "rank": self.rank,
            "score": self.score,
            "accuracy": self.accuracy,
            "avg_confidence": self.avg_confidence,
            "decision_count": self.decision_count,
            "trend": self.trend.value,
            "rank_change": self.rank_change,
            "metrics": self.metrics,
        }


# Confidence level mapping
CONFIDENCE_MAP = {
    "very_low": 0.1,
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7,
    "very_high": 0.9,
}

# Score component weights
SCORE_WEIGHTS = {
    "accuracy": 0.40,  # 40% weight on accuracy
    "confidence": 0.25,  # 25% weight on confidence
    "calibration": 0.20,  # 20% weight on calibration
    "activity": 0.15,  # 15% weight on activity level
}

# Paths
DECISION_DIR = Path(".claude/state/decisions")
LEADERBOARD_DIR = Path(".claude/state/leaderboard")


def calculate_composite_score(
    accuracy: float,
    confidence: float,
    calibration_error: float,
    decision_count: int,
    max_decisions: int = 100,
) -> float:
    """Calculate composite performance score.

    Args:
        accuracy: Accuracy rate (0-1).
        confidence: Average confidence (0-1).
        calibration_error: Absolute difference between confidence and accuracy.
        decision_count: Number of decisions made.
        max_decisions: Maximum expected decisions for normalization.

    Returns:
        Composite score from 0-100.
    """
    # Normalize activity (cap at max)
    activity_score = min(decision_count / max_decisions, 1.0)

    # Calibration score (lower error = higher score)
    calibration_score = max(0, 1.0 - calibration_error * 2)

    # Weighted sum
    score = (
        accuracy * SCORE_WEIGHTS["accuracy"]
        + confidence * SCORE_WEIGHTS["confidence"]
        + calibration_score * SCORE_WEIGHTS["calibration"]
        + activity_score * SCORE_WEIGHTS["activity"]
    ) * 100

    return round(score, 2)


def load_agent_decisions(days: int = 30) -> dict[str, list[dict[str, Any]]]:
    """Load decisions grouped by agent.

    Args:
        days: Number of days to look back.

    Returns:
        Dictionary mapping agent names to decision lists.
    """
    since = datetime.now() - timedelta(days=days)
    agent_decisions: dict[str, list[dict[str, Any]]] = {}

    if not DECISION_DIR.exists():
        return agent_decisions

    for log_file in DECISION_DIR.glob("*.json"):
        try:
            data = json.loads(log_file.read_text())
            timestamp_str = data.get("timestamp", "")

            try:
                decision_time = datetime.fromisoformat(timestamp_str)
                if decision_time < since:
                    continue
            except ValueError:
                continue

            agent = data.get("agent_name", "unknown")
            if agent not in agent_decisions:
                agent_decisions[agent] = []
            agent_decisions[agent].append(data)
        except (json.JSONDecodeError, OSError):
            continue

    return agent_decisions


def calculate_agent_metrics(decisions: list[dict[str, Any]]) -> dict[str, float]:
    """Calculate performance metrics for an agent's decisions.

    Args:
        decisions: List of decision records.

    Returns:
        Dictionary of metric values.
    """
    if not decisions:
        return {
            "accuracy": 0.0,
            "avg_confidence": 0.0,
            "calibration_error": 1.0,
            "decision_count": 0,
        }

    # Extract confidences
    confidences = []
    for d in decisions:
        conf = d.get("confidence")
        if isinstance(conf, float):
            confidences.append(conf)
        else:
            conf_str = d.get("overall_confidence", "medium")
            confidences.append(CONFIDENCE_MAP.get(conf_str, 0.5))

    avg_confidence = statistics.mean(confidences) if confidences else 0.5

    # Calculate accuracy from outcomes
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

    # Calibration error (how well confidence matches accuracy)
    calibration_error = abs(accuracy - avg_confidence)

    return {
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "calibration_error": calibration_error,
        "decision_count": len(decisions),
    }


def load_previous_rankings() -> dict[str, int]:
    """Load previous rankings for trend comparison.

    Returns:
        Dictionary mapping agent names to previous ranks.
    """
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    prev_file = LEADERBOARD_DIR / "previous.json"

    if not prev_file.exists():
        return {}

    try:
        data = json.loads(prev_file.read_text())
        return {r["agent_name"]: r["rank"] for r in data.get("rankings", [])}
    except (json.JSONDecodeError, OSError, KeyError):
        return {}


def save_current_rankings(rankings: list[AgentRanking]) -> None:
    """Save current rankings for future trend comparison.

    Args:
        rankings: Current agent rankings.
    """
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    output_file = LEADERBOARD_DIR / "previous.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "rankings": [{"agent_name": r.agent_name, "rank": r.rank} for r in rankings],
    }
    output_file.write_text(json.dumps(data, indent=2))


def get_leaderboard(days: int = 30) -> list[AgentRanking]:
    """Generate agent performance leaderboard.

    Args:
        days: Number of days to analyze.

    Returns:
        List of AgentRanking objects sorted by score.
    """
    # Load decisions
    agent_decisions = load_agent_decisions(days)

    if not agent_decisions:
        return []

    # Calculate metrics and scores
    agent_data = []
    for agent, decisions in agent_decisions.items():
        metrics = calculate_agent_metrics(decisions)
        score = calculate_composite_score(
            accuracy=metrics["accuracy"],
            confidence=metrics["avg_confidence"],
            calibration_error=metrics["calibration_error"],
            decision_count=int(metrics["decision_count"]),
        )
        agent_data.append(
            {
                "agent": agent,
                "score": score,
                "metrics": metrics,
            }
        )

    # Sort by score (highest first)
    agent_data.sort(key=lambda x: x["score"], reverse=True)

    # Load previous rankings for trend
    prev_rankings = load_previous_rankings()

    # Build ranking objects
    rankings = []
    for i, data in enumerate(agent_data, 1):
        prev_rank = prev_rankings.get(data["agent"], i)
        rank_change = prev_rank - i

        if rank_change > 0:
            trend = TrendDirection.UP
        elif rank_change < 0:
            trend = TrendDirection.DOWN
        else:
            trend = TrendDirection.STABLE

        rankings.append(
            AgentRanking(
                agent_name=data["agent"],
                rank=i,
                score=data["score"],
                accuracy=data["metrics"]["accuracy"],
                avg_confidence=data["metrics"]["avg_confidence"],
                decision_count=int(data["metrics"]["decision_count"]),
                trend=trend,
                rank_change=abs(rank_change),
                metrics=data["metrics"],
            )
        )

    # Save current rankings
    save_current_rankings(rankings)

    return rankings


def print_leaderboard(days: int = 30) -> None:
    """Print formatted leaderboard to console.

    Args:
        days: Number of days to analyze.
    """
    rankings = get_leaderboard(days)

    print("\n" + "=" * 75)
    print("🏆 AGENT PERFORMANCE LEADERBOARD")
    print(f"   Period: Last {days} days | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 75)

    if not rankings:
        print("\n  No agent decisions found for this period.")
        print("=" * 75)
        return

    # Header
    print(f"\n{'Rank':<6}{'Agent':<25}{'Score':<10}{'Accuracy':<10}{'Decisions':<12}{'Trend':<10}")
    print("-" * 75)

    # Rows
    for r in rankings:
        # Trend icons
        if r.trend == TrendDirection.UP:
            trend_str = f"📈 +{r.rank_change}"
        elif r.trend == TrendDirection.DOWN:
            trend_str = f"📉 -{r.rank_change}"
        else:
            trend_str = "➡️  0"

        # Medal for top 3
        rank_str = str(r.rank)
        if r.rank == 1:
            rank_str = "🥇 1"
        elif r.rank == 2:
            rank_str = "🥈 2"
        elif r.rank == 3:
            rank_str = "🥉 3"

        print(
            f"{rank_str:<6}{r.agent_name:<25}{r.score:<10.1f}"
            f"{r.accuracy:<10.1%}{r.decision_count:<12}{trend_str:<10}"
        )

    print("-" * 75)

    # Summary
    total_decisions = sum(r.decision_count for r in rankings)
    avg_score = statistics.mean(r.score for r in rankings) if rankings else 0

    print(f"\nTotal Agents: {len(rankings)} | Total Decisions: {total_decisions} | Avg Score: {avg_score:.1f}")
    print("=" * 75 + "\n")


def main() -> int:
    """Main entry point for leaderboard display.

    Returns:
        Exit code (0 for success).
    """
    import sys

    days = 30
    if "--days" in sys.argv:
        idx = sys.argv.index("--days")
        if idx + 1 < len(sys.argv):
            try:
                days = int(sys.argv[idx + 1])
            except ValueError:
                pass

    if "--json" in sys.argv:
        rankings = get_leaderboard(days)
        print(json.dumps([r.to_dict() for r in rankings], indent=2))
    else:
        print_leaderboard(days)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
