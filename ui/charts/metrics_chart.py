"""
Metrics Chart Widget

Provides agent metrics trend visualization over time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from .base_chart import MATPLOTLIB_AVAILABLE, BaseChartWidget


logger = logging.getLogger(__name__)

# Import matplotlib date formatting if available
mdates = None
if MATPLOTLIB_AVAILABLE:
    try:
        import matplotlib.dates as mdates
    except ImportError:
        pass


@dataclass
class MetricsDataPoint:
    """Single metrics data point.

    Attributes:
        timestamp: When the measurement was taken
        accuracy: Prediction accuracy (0.0 to 1.0)
        confidence: Average confidence level (0.0 to 1.0)
        calibration_error: Expected calibration error (lower is better)
    """

    timestamp: datetime
    accuracy: float
    confidence: float
    calibration_error: float = 0.0


class MetricsChartWidget(BaseChartWidget):
    """Chart widget for visualizing agent metrics over time.

    Displays trend lines for accuracy, confidence, and optionally
    calibration error as agents make predictions.

    Example:
        >>> chart = MetricsChartWidget()
        >>> chart.set_data([
        ...     MetricsDataPoint(datetime.now(), 0.75, 0.80, 0.05),
        ...     MetricsDataPoint(datetime.now(), 0.78, 0.82, 0.04),
        ... ])
    """

    def __init__(
        self,
        show_calibration: bool = False,
        parent: BaseChartWidget | None = None,
    ):
        """Initialize metrics chart.

        Args:
            show_calibration: Whether to show calibration error line
            parent: Parent widget
        """
        super().__init__(title="Agent Metrics Trend", figsize=(8, 4), parent=parent)
        self._data: list[MetricsDataPoint] = []
        self._show_calibration = show_calibration
        self._colors = {
            "accuracy": "#2196F3",  # Blue
            "confidence": "#4CAF50",  # Green
            "calibration": "#FF9800",  # Orange
        }

    @property
    def data(self) -> list[MetricsDataPoint]:
        """Get current metrics data."""
        return self._data.copy()

    @property
    def show_calibration(self) -> bool:
        """Whether calibration error is shown."""
        return self._show_calibration

    @show_calibration.setter
    def show_calibration(self, value: bool) -> None:
        """Set calibration visibility and update chart."""
        self._show_calibration = value
        self._update_chart()

    def set_data(self, data: list[MetricsDataPoint]) -> None:
        """Set metrics data and update chart.

        Args:
            data: List of metrics data points
        """
        self._data = list(data)
        self._update_chart()

    def add_point(self, point: MetricsDataPoint) -> None:
        """Add a single data point and update chart.

        Args:
            point: Metrics data point to add
        """
        self._data.append(point)
        self._update_chart()

    def add_points(self, points: list[MetricsDataPoint]) -> None:
        """Add multiple data points at once.

        Args:
            points: List of data points to add
        """
        self._data.extend(points)
        self._update_chart()

    def clear_data(self) -> None:
        """Clear all data points."""
        self._data.clear()
        self.clear()
        self.refresh()

    def get_latest(self) -> MetricsDataPoint | None:
        """Get the most recent data point.

        Returns:
            Latest data point or None if no data
        """
        return self._data[-1] if self._data else None

    def get_averages(self) -> tuple[float, float, float]:
        """Calculate average metrics.

        Returns:
            Tuple of (avg_accuracy, avg_confidence, avg_calibration_error)
        """
        if not self._data:
            return (0.0, 0.0, 0.0)

        n = len(self._data)
        avg_acc = sum(p.accuracy for p in self._data) / n
        avg_conf = sum(p.confidence for p in self._data) / n
        avg_cal = sum(p.calibration_error for p in self._data) / n
        return (avg_acc, avg_conf, avg_cal)

    def _update_chart(self) -> None:
        """Redraw the chart with current data."""
        if not self.is_available():
            return

        self.clear()

        if not self._data:
            self.ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=12,
                color="gray",
            )
            self.refresh()
            return

        # Extract data
        timestamps = [p.timestamp for p in self._data]
        accuracies = [p.accuracy * 100 for p in self._data]
        confidences = [p.confidence * 100 for p in self._data]

        # Plot accuracy line
        self.ax.plot(
            timestamps,
            accuracies,
            color=self._colors["accuracy"],
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Accuracy %",
        )

        # Plot confidence line
        self.ax.plot(
            timestamps,
            confidences,
            color=self._colors["confidence"],
            linestyle="--",
            linewidth=2,
            marker="s",
            markersize=4,
            label="Confidence %",
        )

        # Optionally plot calibration error
        if self._show_calibration:
            calibrations = [p.calibration_error * 100 for p in self._data]
            self.ax.plot(
                timestamps,
                calibrations,
                color=self._colors["calibration"],
                linestyle=":",
                linewidth=2,
                marker="^",
                markersize=4,
                label="Calibration Error %",
            )

        # Configure axes
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Percentage")
        self.ax.set_ylim(0, 105)  # Slightly above 100 for visual clarity
        self.ax.legend(loc="lower right")
        self.ax.grid(True, alpha=0.3)

        # Format x-axis with dates
        if mdates is not None and len(timestamps) > 1:
            time_range = (timestamps[-1] - timestamps[0]).total_seconds()
            if time_range < 3600:  # Less than 1 hour
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            elif time_range < 86400:  # Less than 1 day
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            else:
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))

            self.figure.autofmt_xdate()

        # Add average lines if enough data
        if len(self._data) >= 5:
            avg_acc, avg_conf, _ = self.get_averages()
            self.ax.axhline(y=avg_acc * 100, color=self._colors["accuracy"], linestyle="-.", alpha=0.5, linewidth=1)
            self.ax.axhline(y=avg_conf * 100, color=self._colors["confidence"], linestyle="-.", alpha=0.5, linewidth=1)

        self.refresh()


def create_metrics_chart(
    show_calibration: bool = False,
    parent: BaseChartWidget | None = None,
) -> MetricsChartWidget:
    """Factory function to create metrics chart.

    Args:
        show_calibration: Whether to show calibration error
        parent: Parent widget

    Returns:
        MetricsChartWidget instance
    """
    return MetricsChartWidget(show_calibration=show_calibration, parent=parent)
