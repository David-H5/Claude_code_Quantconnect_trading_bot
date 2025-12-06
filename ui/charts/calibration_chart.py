"""
Calibration Chart Widget

Provides confidence calibration reliability diagrams.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .base_chart import MATPLOTLIB_AVAILABLE, BaseChartWidget


logger = logging.getLogger(__name__)

# Import numpy if available for binning
np = None
if MATPLOTLIB_AVAILABLE:
    try:
        import numpy as np
    except ImportError:
        pass


@dataclass
class CalibrationBin:
    """Calibration bin data.

    Represents a bin in a calibration histogram, tracking predicted
    confidence ranges and actual accuracy within that range.

    Attributes:
        confidence_range: (low, high) confidence bounds for this bin
        mean_confidence: Average confidence of predictions in this bin
        accuracy: Fraction of correct predictions in this bin
        count: Number of predictions in this bin
    """

    confidence_range: tuple[float, float]
    mean_confidence: float
    accuracy: float
    count: int


class CalibrationChartWidget(BaseChartWidget):
    """Reliability diagram for confidence calibration.

    A calibration chart (reliability diagram) compares predicted confidence
    levels against actual accuracy. A well-calibrated model has bars that
    follow the diagonal line.

    Example:
        >>> chart = CalibrationChartWidget()
        >>> chart.compute_from_predictions(
        ...     confidences=[0.7, 0.8, 0.9, 0.6],
        ...     correct=[True, True, False, True],
        ... )
    """

    def __init__(
        self,
        n_bins: int = 10,
        show_histogram: bool = True,
        parent: BaseChartWidget | None = None,
    ):
        """Initialize calibration chart.

        Args:
            n_bins: Number of confidence bins (default 10)
            show_histogram: Whether to show sample count histogram
            parent: Parent widget
        """
        super().__init__(title="Confidence Calibration", figsize=(5, 5), parent=parent)
        self._bins: list[CalibrationBin] = []
        self._n_bins = n_bins
        self._show_histogram = show_histogram
        self._expected_calibration_error: float = 0.0

    @property
    def bins(self) -> list[CalibrationBin]:
        """Get current calibration bins."""
        return self._bins.copy()

    @property
    def expected_calibration_error(self) -> float:
        """Get Expected Calibration Error (ECE).

        ECE measures the weighted average of calibration gaps across bins.
        Lower values indicate better calibration.
        """
        return self._expected_calibration_error

    @property
    def n_bins(self) -> int:
        """Get number of bins."""
        return self._n_bins

    @n_bins.setter
    def n_bins(self, value: int) -> None:
        """Set number of bins and recalculate if data exists."""
        self._n_bins = max(2, min(20, value))  # Clamp between 2 and 20
        if self._bins:
            self._update_chart()

    def set_data(self, bins: list[CalibrationBin]) -> None:
        """Set calibration bins directly and update chart.

        Args:
            bins: List of calibration bins
        """
        self._bins = list(bins)
        self._compute_ece()
        self._update_chart()

    def compute_from_predictions(
        self,
        confidences: list[float],
        correct: list[bool],
        n_bins: int | None = None,
    ) -> None:
        """Compute calibration from raw predictions.

        Args:
            confidences: List of predicted confidence values (0.0 to 1.0)
            correct: List of boolean outcomes (True = correct prediction)
            n_bins: Number of bins (uses instance default if not specified)
        """
        if not confidences or len(confidences) != len(correct):
            logger.warning("Invalid input for calibration computation")
            return

        n_bins = n_bins or self._n_bins
        bins: list[CalibrationBin] = []

        # Compute bin edges
        if np is not None:
            bin_edges = np.linspace(0, 1, n_bins + 1)
        else:
            bin_edges = [i / n_bins for i in range(n_bins + 1)]

        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]

            # Find predictions in this bin
            bin_confidences = []
            bin_correct = []
            for conf, corr in zip(confidences, correct):
                # Include upper edge only for last bin
                if i == n_bins - 1:
                    in_bin = low <= conf <= high
                else:
                    in_bin = low <= conf < high

                if in_bin:
                    bin_confidences.append(conf)
                    bin_correct.append(corr)

            if bin_confidences:
                mean_conf = sum(bin_confidences) / len(bin_confidences)
                acc = sum(1 for c in bin_correct if c) / len(bin_correct)
                bins.append(
                    CalibrationBin(
                        confidence_range=(low, high),
                        mean_confidence=mean_conf,
                        accuracy=acc,
                        count=len(bin_confidences),
                    )
                )

        self._bins = bins
        self._compute_ece()
        self._update_chart()

    def _compute_ece(self) -> None:
        """Compute Expected Calibration Error."""
        if not self._bins:
            self._expected_calibration_error = 0.0
            return

        total_samples = sum(b.count for b in self._bins)
        if total_samples == 0:
            self._expected_calibration_error = 0.0
            return

        ece = sum(b.count * abs(b.accuracy - b.mean_confidence) for b in self._bins) / total_samples

        self._expected_calibration_error = ece

    def _update_chart(self) -> None:
        """Redraw the calibration diagram."""
        if not self.is_available():
            return

        self.clear()

        # Perfect calibration line
        self.ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")

        if not self._bins:
            self.ax.text(
                0.5,
                0.5,
                "No calibration data",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=12,
                color="gray",
            )
            self.refresh()
            return

        # Extract data
        confidences = [b.mean_confidence for b in self._bins]
        accuracies = [b.accuracy for b in self._bins]
        counts = [b.count for b in self._bins]

        # Bar width based on bin size
        width = 0.9 / self._n_bins

        # Plot calibration bars
        self.ax.bar(
            confidences,
            accuracies,
            width=width,
            alpha=0.7,
            color="steelblue",
            edgecolor="navy",
            label="Model Calibration",
        )

        # Add gap visualization (shaded area between bars and diagonal)
        for conf, acc in zip(confidences, accuracies):
            gap_color = "red" if acc < conf else "green"
            self.ax.plot([conf, conf], [min(acc, conf), max(acc, conf)], color=gap_color, linewidth=2, alpha=0.5)

        # Configure axes
        self.ax.set_xlabel("Mean Predicted Confidence")
        self.ax.set_ylabel("Fraction of Correct Predictions")
        self.ax.set_xlim(-0.05, 1.05)
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.set_aspect("equal")
        self.ax.legend(loc="lower right")
        self.ax.grid(True, alpha=0.3)

        # Add ECE annotation
        self.ax.annotate(
            f"ECE: {self._expected_calibration_error:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Add sample count histogram on secondary axis if enabled
        if self._show_histogram and counts:
            ax2 = self.ax.twinx()
            ax2.bar(confidences, counts, width=width * 0.4, alpha=0.3, color="gray", label="Sample Count")
            ax2.set_ylabel("Sample Count", color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")
            ax2.set_ylim(0, max(counts) * 3)  # Leave room for main chart

        self.refresh()

    def get_reliability_summary(self) -> dict:
        """Get summary of calibration reliability.

        Returns:
            Dictionary with calibration metrics
        """
        if not self._bins:
            return {
                "ece": 0.0,
                "total_samples": 0,
                "num_bins": 0,
                "is_overconfident": False,
                "is_underconfident": False,
            }

        total_samples = sum(b.count for b in self._bins)
        overconfident_count = sum(b.count for b in self._bins if b.mean_confidence > b.accuracy)
        underconfident_count = sum(b.count for b in self._bins if b.mean_confidence < b.accuracy)

        return {
            "ece": self._expected_calibration_error,
            "total_samples": total_samples,
            "num_bins": len(self._bins),
            "is_overconfident": overconfident_count > underconfident_count,
            "is_underconfident": underconfident_count > overconfident_count,
            "overconfident_fraction": overconfident_count / total_samples if total_samples else 0,
        }


def create_calibration_chart(
    n_bins: int = 10,
    show_histogram: bool = True,
    parent: BaseChartWidget | None = None,
) -> CalibrationChartWidget:
    """Factory function to create calibration chart.

    Args:
        n_bins: Number of confidence bins
        show_histogram: Whether to show sample count histogram
        parent: Parent widget

    Returns:
        CalibrationChartWidget instance
    """
    return CalibrationChartWidget(n_bins=n_bins, show_histogram=show_histogram, parent=parent)
