"""
ELO History Chart Widget (UPGRADE-010 Sprint 2 P1)

Visualizes agent ELO rating progression over time.
Supports multi-agent comparison and time range selection.

QuantConnect Compatible: N/A (UI component)
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta
from typing import Any

from .base_chart import (
    MATPLOTLIB_AVAILABLE,
    BaseChartWidget,
)


logger = logging.getLogger(__name__)


class ELOHistoryChart(BaseChartWidget):
    """
    Chart widget for visualizing ELO rating history.

    Features:
    - Line chart showing ELO over time
    - Multi-agent comparison
    - Time range selection
    - Export to PNG/CSV
    """

    def __init__(
        self,
        parent: Any | None = None,
        title: str = "ELO History",
        figsize: tuple[float, float] = (10, 6),
    ):
        """
        Initialize ELO history chart.

        Args:
            parent: Parent widget
            title: Chart title
            figsize: Figure size (width, height)
        """
        super().__init__(parent=parent, title=title, figsize=figsize)
        self._history_data: dict[str, list[dict[str, Any]]] = {}
        self._colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        self._time_range_days: int | None = None  # None = all time

    def set_data(self, history_data: dict[str, list[dict[str, Any]]]) -> None:
        """
        Set ELO history data.

        Args:
            history_data: Dictionary mapping agent names to history entries
                Each entry: {"timestamp": str, "elo_rating": float, "change": float}
        """
        self._history_data = history_data
        self.update_chart()

    def set_time_range(self, days: int | None) -> None:
        """
        Set time range filter.

        Args:
            days: Number of days to show (None = all time)
        """
        self._time_range_days = days
        self.update_chart()

    def update_chart(self) -> None:
        """Redraw the chart with current data."""
        if not MATPLOTLIB_AVAILABLE:
            self._show_placeholder("matplotlib not available")
            return

        if not self._history_data:
            self._show_placeholder("No ELO history data")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Calculate time filter if set
        cutoff_time: datetime | None = None
        if self._time_range_days is not None:
            cutoff_time = datetime.utcnow() - timedelta(days=self._time_range_days)

        # Plot each agent's history
        for idx, (agent_name, history) in enumerate(self._history_data.items()):
            if not history:
                continue

            # Parse and filter data
            times: list[datetime] = []
            ratings: list[float] = []

            for entry in history:
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if cutoff_time and timestamp < cutoff_time:
                    continue
                times.append(timestamp)
                ratings.append(entry["elo_rating"])

            if not times:
                continue

            color = self._colors[idx % len(self._colors)]
            ax.plot(
                times,
                ratings,
                label=agent_name,
                color=color,
                linewidth=2,
                marker="o",
                markersize=3,
                alpha=0.8,
            )

        # Formatting
        ax.set_xlabel("Time")
        ax.set_ylabel("ELO Rating")
        ax.set_title(self._title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add baseline reference
        ax.axhline(y=1500, color="gray", linestyle="--", alpha=0.5, label="Baseline (1500)")

        # Rotate x-axis labels for readability
        self.figure.autofmt_xdate()

        self.figure.tight_layout()
        self.canvas.draw()

    def export_csv(self) -> str:
        """
        Export data to CSV format.

        Returns:
            CSV string
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["agent_name", "timestamp", "elo_rating", "change", "outcome"])

        # Data rows
        for agent_name, history in self._history_data.items():
            for entry in history:
                writer.writerow(
                    [
                        agent_name,
                        entry.get("timestamp", ""),
                        entry.get("elo_rating", ""),
                        entry.get("change", ""),
                        entry.get("outcome", ""),
                    ]
                )

        return output.getvalue()

    def export_png(self, filepath: str) -> bool:
        """
        Export chart to PNG file.

        Args:
            filepath: Output file path

        Returns:
            True if successful
        """
        if not MATPLOTLIB_AVAILABLE:
            return False

        try:
            self.figure.savefig(filepath, dpi=150, bbox_inches="tight")
            return True
        except Exception as e:
            logger.error(f"Failed to export PNG: {e}")
            return False

    def get_agent_summary(self) -> dict[str, dict[str, Any]]:
        """
        Get summary statistics for each agent.

        Returns:
            Dictionary with agent summaries
        """
        summaries: dict[str, dict[str, Any]] = {}

        for agent_name, history in self._history_data.items():
            if not history:
                continue

            ratings = [e["elo_rating"] for e in history]
            changes = [e["change"] for e in history]

            summaries[agent_name] = {
                "current_elo": ratings[-1] if ratings else 1500,
                "highest_elo": max(ratings) if ratings else 1500,
                "lowest_elo": min(ratings) if ratings else 1500,
                "total_changes": len(history),
                "avg_change": sum(changes) / len(changes) if changes else 0,
                "total_change": sum(changes) if changes else 0,
            }

        return summaries


def create_elo_history_chart(
    parent: Any | None = None,
    title: str = "ELO History",
    figsize: tuple[float, float] = (10, 6),
) -> ELOHistoryChart:
    """
    Factory function to create an ELO history chart.

    Args:
        parent: Parent widget
        title: Chart title
        figsize: Figure size

    Returns:
        ELOHistoryChart instance
    """
    return ELOHistoryChart(parent=parent, title=title, figsize=figsize)
