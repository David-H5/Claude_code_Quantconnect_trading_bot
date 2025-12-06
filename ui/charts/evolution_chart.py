"""
Evolution Chart Widget

Provides evolution progress visualization across cycles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .base_chart import BaseChartWidget


logger = logging.getLogger(__name__)


@dataclass
class EvolutionCycle:
    """Single evolution cycle data.

    Attributes:
        cycle_number: Sequential cycle number (1-indexed)
        score: Performance score for this cycle (0.0 to 1.0)
        improvement: Score change from previous cycle
        prompt_version: Version string of the prompt used
    """

    cycle_number: int
    score: float
    improvement: float = 0.0
    prompt_version: str = ""


class EvolutionChartWidget(BaseChartWidget):
    """Chart showing agent evolution progress over cycles.

    Displays score trends and improvement bars across evolution cycles,
    helping visualize whether agents are improving over time.

    Example:
        >>> chart = EvolutionChartWidget()
        >>> chart.set_data([
        ...     EvolutionCycle(1, 0.65, 0.0, "v1.0"),
        ...     EvolutionCycle(2, 0.72, 0.07, "v1.1"),
        ...     EvolutionCycle(3, 0.75, 0.03, "v1.2"),
        ... ])
    """

    def __init__(
        self,
        show_improvement_bars: bool = True,
        parent: BaseChartWidget | None = None,
    ):
        """Initialize evolution chart.

        Args:
            show_improvement_bars: Whether to show improvement bars
            parent: Parent widget
        """
        super().__init__(title="Evolution Progress", figsize=(8, 4), parent=parent)
        self._cycles: list[EvolutionCycle] = []
        self._show_improvement_bars = show_improvement_bars
        self._colors = {
            "score": "#2196F3",
            "improvement_positive": "#4CAF50",
            "improvement_negative": "#F44336",
            "baseline": "#9E9E9E",
        }

    @property
    def cycles(self) -> list[EvolutionCycle]:
        """Get current evolution cycles."""
        return self._cycles.copy()

    @property
    def total_improvement(self) -> float:
        """Calculate total score improvement from first to last cycle."""
        if len(self._cycles) < 2:
            return 0.0
        return self._cycles[-1].score - self._cycles[0].score

    @property
    def best_cycle(self) -> EvolutionCycle | None:
        """Get the cycle with highest score."""
        if not self._cycles:
            return None
        return max(self._cycles, key=lambda c: c.score)

    def set_data(self, cycles: list[EvolutionCycle]) -> None:
        """Set evolution cycle data.

        Args:
            cycles: List of evolution cycles
        """
        self._cycles = list(cycles)
        self._update_chart()

    def add_cycle(self, cycle: EvolutionCycle) -> None:
        """Add a single cycle.

        If improvement is not set, it will be calculated from previous cycle.

        Args:
            cycle: Evolution cycle to add
        """
        if self._cycles and cycle.improvement == 0.0:
            # Calculate improvement if not provided
            prev_score = self._cycles[-1].score
            cycle = EvolutionCycle(
                cycle_number=cycle.cycle_number,
                score=cycle.score,
                improvement=cycle.score - prev_score,
                prompt_version=cycle.prompt_version,
            )
        self._cycles.append(cycle)
        self._update_chart()

    def clear_data(self) -> None:
        """Clear all cycle data."""
        self._cycles.clear()
        self.clear()
        self.refresh()

    def get_convergence_status(self) -> tuple[bool, str]:
        """Check if evolution has converged.

        Returns:
            Tuple of (is_converged, reason)
        """
        if len(self._cycles) < 3:
            return (False, "Not enough cycles")

        # Check last 3 cycles for minimal improvement
        recent = self._cycles[-3:]
        recent_improvements = [abs(c.improvement) for c in recent]

        if all(imp < 0.01 for imp in recent_improvements):
            return (True, "Improvements below threshold for 3 cycles")

        # Check for declining trend
        if all(c.improvement < 0 for c in recent):
            return (True, "Score declining for 3 cycles")

        return (False, "Still improving")

    def _update_chart(self) -> None:
        """Redraw the evolution chart."""
        if not self.is_available():
            return

        self.clear()

        if not self._cycles:
            self.ax.text(
                0.5,
                0.5,
                "No evolution data",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=12,
                color="gray",
            )
            self.refresh()
            return

        cycle_nums = [c.cycle_number for c in self._cycles]
        scores = [c.score * 100 for c in self._cycles]
        improvements = [c.improvement * 100 for c in self._cycles]

        # Plot score line
        self.ax.plot(
            cycle_nums,
            scores,
            color=self._colors["score"],
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=2,
            label="Score %",
        )

        # Add baseline from first cycle
        if len(self._cycles) > 1:
            baseline = self._cycles[0].score * 100
            self.ax.axhline(
                y=baseline,
                color=self._colors["baseline"],
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=f"Baseline ({baseline:.1f}%)",
            )

        # Improvement bars on secondary axis
        if self._show_improvement_bars and len(self._cycles) > 1:
            ax2 = self.ax.twinx()
            colors = [
                self._colors["improvement_positive"] if imp >= 0 else self._colors["improvement_negative"]
                for imp in improvements
            ]

            bar_width = 0.4
            ax2.bar(cycle_nums, improvements, width=bar_width, alpha=0.4, color=colors, label="Improvement %")
            ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
            ax2.set_ylabel("Improvement %", color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")

            # Auto-scale improvement axis
            max_imp = max(abs(i) for i in improvements) if improvements else 5
            ax2.set_ylim(-max_imp * 1.5, max_imp * 1.5)

        # Configure main axes
        self.ax.set_xlabel("Evolution Cycle")
        self.ax.set_ylabel("Score %", color=self._colors["score"])
        self.ax.tick_params(axis="y", labelcolor=self._colors["score"])

        # Set x-axis to integer ticks
        self.ax.set_xticks(cycle_nums)

        # Score axis limits
        min_score = min(scores) - 5
        max_score = max(scores) + 5
        self.ax.set_ylim(max(0, min_score), min(100, max_score))

        self.ax.legend(loc="upper left")
        self.ax.grid(True, alpha=0.3)

        # Mark best score
        if len(self._cycles) > 1:
            best = self.best_cycle
            if best:
                best_idx = cycle_nums.index(best.cycle_number)
                self.ax.annotate(
                    f"Best: {best.score * 100:.1f}%",
                    xy=(best.cycle_number, best.score * 100),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=9,
                    color="green",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                )

        # Add total improvement annotation
        total_imp = self.total_improvement * 100
        color = "green" if total_imp >= 0 else "red"
        self.ax.annotate(
            f"Total: {total_imp:+.1f}%",
            xy=(0.98, 0.02),
            xycoords="axes fraction",
            fontsize=10,
            color=color,
            ha="right",
            fontweight="bold",
        )

        self.refresh()

    def get_evolution_summary(self) -> dict:
        """Get summary of evolution progress.

        Returns:
            Dictionary with evolution metrics
        """
        if not self._cycles:
            return {
                "num_cycles": 0,
                "current_score": 0.0,
                "best_score": 0.0,
                "total_improvement": 0.0,
                "is_converged": False,
            }

        is_converged, reason = self.get_convergence_status()
        best = self.best_cycle

        return {
            "num_cycles": len(self._cycles),
            "current_score": self._cycles[-1].score,
            "best_score": best.score if best else 0.0,
            "best_cycle": best.cycle_number if best else 0,
            "total_improvement": self.total_improvement,
            "is_converged": is_converged,
            "convergence_reason": reason,
        }


def create_evolution_chart(
    show_improvement_bars: bool = True,
    parent: BaseChartWidget | None = None,
) -> EvolutionChartWidget:
    """Factory function to create evolution chart.

    Args:
        show_improvement_bars: Whether to show improvement bars
        parent: Parent widget

    Returns:
        EvolutionChartWidget instance
    """
    return EvolutionChartWidget(show_improvement_bars=show_improvement_bars, parent=parent)
