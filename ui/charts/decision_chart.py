"""
Decision Chart Widget

Provides decision distribution visualization with bar and pie charts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from .base_chart import BaseChartWidget


logger = logging.getLogger(__name__)


class DecisionOutcome(Enum):
    """Possible decision outcomes."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    PENDING = "pending"
    SKIPPED = "skipped"


@dataclass
class DecisionStats:
    """Decision statistics by type.

    Attributes:
        decision_type: Name of the decision type (e.g., "BUY", "SELL")
        correct: Number of correct decisions
        incorrect: Number of incorrect decisions
        pending: Number of pending decisions
        skipped: Number of skipped decisions
    """

    decision_type: str
    correct: int = 0
    incorrect: int = 0
    pending: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        """Get total number of decisions."""
        return self.correct + self.incorrect + self.pending + self.skipped

    @property
    def accuracy(self) -> float:
        """Get accuracy (correct / resolved)."""
        resolved = self.correct + self.incorrect
        return self.correct / resolved if resolved > 0 else 0.0


class DecisionChartWidget(BaseChartWidget):
    """Chart showing decision distribution.

    Supports both bar charts (for comparing across decision types) and
    pie charts (for showing overall outcome distribution).

    Example:
        >>> chart = DecisionChartWidget(chart_type="bar")
        >>> chart.set_data([
        ...     DecisionStats("BUY", correct=10, incorrect=3, pending=2),
        ...     DecisionStats("SELL", correct=8, incorrect=5, pending=1),
        ... ])
    """

    def __init__(
        self,
        chart_type: str = "bar",
        parent: BaseChartWidget | None = None,
    ):
        """Initialize decision chart.

        Args:
            chart_type: "bar" for grouped bar chart, "pie" for pie chart
            parent: Parent widget
        """
        super().__init__(title="Decision Distribution", figsize=(7, 4), parent=parent)
        self._chart_type = chart_type
        self._stats: list[DecisionStats] = []
        self._outcome_counts: dict[str, int] = {}
        self._colors = {
            "correct": "#4CAF50",
            "incorrect": "#F44336",
            "pending": "#9E9E9E",
            "skipped": "#FFC107",
        }

    @property
    def chart_type(self) -> str:
        """Get current chart type."""
        return self._chart_type

    @chart_type.setter
    def chart_type(self, value: str) -> None:
        """Set chart type and redraw."""
        if value in ("bar", "pie"):
            self._chart_type = value
            self._update_chart()

    @property
    def stats(self) -> list[DecisionStats]:
        """Get current decision stats."""
        return self._stats.copy()

    def set_data(self, stats: list[DecisionStats]) -> None:
        """Set decision statistics.

        Args:
            stats: List of decision stats by type
        """
        self._stats = list(stats)
        self._update_chart()

    def set_outcome_counts(self, counts: dict[str, int]) -> None:
        """Set simple outcome counts for pie chart.

        Args:
            counts: Dictionary mapping outcome names to counts
        """
        self._outcome_counts = dict(counts)
        if self._chart_type == "pie":
            self._update_chart()

    def add_decision(
        self,
        decision_type: str,
        outcome: DecisionOutcome,
    ) -> None:
        """Record a single decision.

        Args:
            decision_type: Type of decision
            outcome: Outcome of the decision
        """
        # Find or create stats for this type
        stat = next((s for s in self._stats if s.decision_type == decision_type), None)
        if not stat:
            stat = DecisionStats(decision_type=decision_type)
            self._stats.append(stat)

        # Update count
        if outcome == DecisionOutcome.CORRECT:
            stat.correct += 1
        elif outcome == DecisionOutcome.INCORRECT:
            stat.incorrect += 1
        elif outcome == DecisionOutcome.PENDING:
            stat.pending += 1
        elif outcome == DecisionOutcome.SKIPPED:
            stat.skipped += 1

        self._update_chart()

    def clear_data(self) -> None:
        """Clear all decision data."""
        self._stats.clear()
        self._outcome_counts.clear()
        self.clear()
        self.refresh()

    def _update_chart(self) -> None:
        """Redraw the chart."""
        if not self.is_available():
            return

        if self._chart_type == "pie":
            self._update_pie_chart()
        else:
            self._update_bar_chart()

    def _update_bar_chart(self) -> None:
        """Redraw as grouped bar chart."""
        self.clear()

        if not self._stats:
            self.ax.text(
                0.5,
                0.5,
                "No decision data",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=12,
                color="gray",
            )
            self.refresh()
            return

        types = [s.decision_type for s in self._stats]
        correct = [s.correct for s in self._stats]
        incorrect = [s.incorrect for s in self._stats]
        pending = [s.pending for s in self._stats]
        skipped = [s.skipped for s in self._stats]

        x = list(range(len(types)))
        width = 0.2

        # Grouped bars
        self.ax.bar([i - 1.5 * width for i in x], correct, width, label="Correct", color=self._colors["correct"])
        self.ax.bar([i - 0.5 * width for i in x], incorrect, width, label="Incorrect", color=self._colors["incorrect"])
        self.ax.bar([i + 0.5 * width for i in x], pending, width, label="Pending", color=self._colors["pending"])
        self.ax.bar([i + 1.5 * width for i in x], skipped, width, label="Skipped", color=self._colors["skipped"])

        self.ax.set_xlabel("Decision Type")
        self.ax.set_ylabel("Count")
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(types)
        self.ax.legend(loc="upper right")
        self.ax.grid(True, alpha=0.3, axis="y")

        # Rotate labels if many types
        if len(types) > 4:
            self.figure.autofmt_xdate()

        # Add accuracy annotations
        for i, stat in enumerate(self._stats):
            if stat.total > 0:
                acc = stat.accuracy * 100
                self.ax.annotate(
                    f"{acc:.0f}%",
                    xy=(i, max(stat.correct, stat.incorrect, stat.pending, stat.skipped) + 1),
                    ha="center",
                    fontsize=8,
                    color="black",
                )

        self.refresh()

    def _update_pie_chart(self) -> None:
        """Redraw as pie chart."""
        self.clear()

        # Use outcome counts if set, otherwise aggregate from stats
        if self._outcome_counts:
            labels = list(self._outcome_counts.keys())
            sizes = list(self._outcome_counts.values())
        elif self._stats:
            labels = ["Correct", "Incorrect", "Pending", "Skipped"]
            sizes = [
                sum(s.correct for s in self._stats),
                sum(s.incorrect for s in self._stats),
                sum(s.pending for s in self._stats),
                sum(s.skipped for s in self._stats),
            ]
            # Remove zero values
            non_zero = [(l, s) for l, s in zip(labels, sizes) if s > 0]
            if non_zero:
                labels, sizes = zip(*non_zero)
                labels, sizes = list(labels), list(sizes)
            else:
                labels, sizes = [], []
        else:
            self.ax.text(
                0.5,
                0.5,
                "No decision data",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=12,
                color="gray",
            )
            self.refresh()
            return

        if not sizes or sum(sizes) == 0:
            self.ax.text(
                0.5,
                0.5,
                "No decisions recorded",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=12,
                color="gray",
            )
            self.refresh()
            return

        # Assign colors
        pie_colors = [self._colors.get(l.lower(), "#2196F3") for l in labels]

        wedges, texts, autotexts = self.ax.pie(
            sizes,
            labels=labels,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=[0.02] * len(sizes),  # Slight separation
        )

        # Style percentage labels
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight("bold")

        self.ax.axis("equal")

        # Add total count in center
        total = sum(sizes)
        self.ax.annotate(f"Total\n{total}", xy=(0, 0), ha="center", va="center", fontsize=12, fontweight="bold")

        self.refresh()

    def get_decision_summary(self) -> dict:
        """Get summary of decision distribution.

        Returns:
            Dictionary with decision metrics
        """
        if not self._stats:
            return {
                "total_decisions": 0,
                "overall_accuracy": 0.0,
                "by_type": {},
            }

        total_correct = sum(s.correct for s in self._stats)
        total_incorrect = sum(s.incorrect for s in self._stats)
        total_resolved = total_correct + total_incorrect

        return {
            "total_decisions": sum(s.total for s in self._stats),
            "overall_accuracy": total_correct / total_resolved if total_resolved else 0.0,
            "by_type": {
                s.decision_type: {
                    "total": s.total,
                    "accuracy": s.accuracy,
                    "correct": s.correct,
                    "incorrect": s.incorrect,
                }
                for s in self._stats
            },
        }


def create_decision_chart(
    chart_type: str = "bar",
    parent: BaseChartWidget | None = None,
) -> DecisionChartWidget:
    """Factory function to create decision chart.

    Args:
        chart_type: "bar" or "pie"
        parent: Parent widget

    Returns:
        DecisionChartWidget instance
    """
    return DecisionChartWidget(chart_type=chart_type, parent=parent)
