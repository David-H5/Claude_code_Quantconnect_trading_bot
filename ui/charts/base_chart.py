"""
Base Chart Widget

Provides base chart widget with matplotlib/PySide6 integration.
Falls back gracefully when matplotlib is not available.
"""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)

# Check matplotlib availability
MATPLOTLIB_AVAILABLE = False
FigureCanvasQTAgg: Any = None
Figure: Any = None

try:
    import matplotlib

    matplotlib.use("Qt5Agg")  # Use Qt backend
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvas
    from matplotlib.figure import Figure as _Figure

    FigureCanvasQTAgg = _FigureCanvas
    Figure = _Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("matplotlib not available - charts will show placeholders")

# PySide6 imports with fallback
try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False

    class QWidget:
        """Stub QWidget for when PySide6 is not available."""

        def __init__(self, parent=None):
            self._parent = parent
            self._layout = None

        def setLayout(self, layout):
            self._layout = layout

    class QVBoxLayout:
        """Stub QVBoxLayout."""

        def __init__(self, parent=None):
            self._widgets = []

        def setContentsMargins(self, *args):
            pass

        def addWidget(self, widget):
            self._widgets.append(widget)

    class QLabel:
        """Stub QLabel."""

        def __init__(self, text=""):
            self._text = text

        def setAlignment(self, alignment):
            pass

        def setText(self, text):
            self._text = text

        def text(self):
            return self._text

    class QSizePolicy:
        """Stub QSizePolicy."""

        Expanding = 0
        Preferred = 0

    class Qt:
        """Stub Qt namespace."""

        AlignCenter = 0


class BaseChartWidget(QWidget):
    """Base widget for matplotlib charts with graceful fallback.

    This widget provides a consistent interface for displaying matplotlib
    charts within PySide6 applications. When matplotlib is not available,
    it shows a placeholder message.

    Attributes:
        title: Chart title displayed at the top
        figsize: Figure size in inches (width, height)
        figure: Matplotlib Figure object (None if unavailable)
        canvas: Qt canvas for matplotlib (None if unavailable)
        ax: Main axes object for plotting (None if unavailable)
    """

    def __init__(
        self,
        title: str = "Chart",
        figsize: tuple[float, float] = (6, 4),
        parent: QWidget | None = None,
    ):
        """Initialize the chart widget.

        Args:
            title: Chart title
            figsize: Figure size as (width, height) in inches
            parent: Parent widget
        """
        if PYSIDE6_AVAILABLE:
            super().__init__(parent)
        else:
            super().__init__(parent=parent)

        self.title = title
        self.figsize = figsize
        self.figure: Any = None
        self.canvas: Any = None
        self.ax: Any = None
        self._placeholder: QLabel | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Initialize the chart UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if MATPLOTLIB_AVAILABLE and Figure is not None and FigureCanvasQTAgg is not None:
            self.figure = Figure(figsize=self.figsize)
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_title(self.title)

            if PYSIDE6_AVAILABLE:
                self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            layout.addWidget(self.canvas)
        else:
            # Fallback placeholder
            self._placeholder = QLabel(f"[{self.title}]\nmatplotlib not available")
            if PYSIDE6_AVAILABLE:
                self._placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(self._placeholder)

        if PYSIDE6_AVAILABLE:
            self.setLayout(layout)

    def clear(self) -> None:
        """Clear the chart axes."""
        if self.ax is not None:
            self.ax.clear()
            self.ax.set_title(self.title)

    def refresh(self) -> None:
        """Redraw the chart canvas."""
        if self.canvas is not None:
            self.figure.tight_layout()
            self.canvas.draw()

    def set_title(self, title: str) -> None:
        """Update chart title.

        Args:
            title: New chart title
        """
        self.title = title
        if self.ax is not None:
            self.ax.set_title(title)
            self.refresh()
        elif self._placeholder is not None:
            self._placeholder.setText(f"[{title}]\nmatplotlib not available")

    def set_xlabel(self, label: str) -> None:
        """Set x-axis label.

        Args:
            label: X-axis label text
        """
        if self.ax is not None:
            self.ax.set_xlabel(label)

    def set_ylabel(self, label: str) -> None:
        """Set y-axis label.

        Args:
            label: Y-axis label text
        """
        if self.ax is not None:
            self.ax.set_ylabel(label)

    def set_xlim(self, left: float, right: float) -> None:
        """Set x-axis limits.

        Args:
            left: Left limit
            right: Right limit
        """
        if self.ax is not None:
            self.ax.set_xlim(left, right)

    def set_ylim(self, bottom: float, top: float) -> None:
        """Set y-axis limits.

        Args:
            bottom: Bottom limit
            top: Top limit
        """
        if self.ax is not None:
            self.ax.set_ylim(bottom, top)

    def grid(self, visible: bool = True, alpha: float = 0.3) -> None:
        """Toggle grid visibility.

        Args:
            visible: Whether grid is visible
            alpha: Grid line transparency
        """
        if self.ax is not None:
            self.ax.grid(visible, alpha=alpha)

    def legend(self, loc: str = "best") -> None:
        """Add legend to chart.

        Args:
            loc: Legend location (e.g., 'best', 'upper right', 'lower left')
        """
        if self.ax is not None:
            self.ax.legend(loc=loc)

    def is_available(self) -> bool:
        """Check if matplotlib charting is available.

        Returns:
            True if matplotlib is available and chart is functional
        """
        return MATPLOTLIB_AVAILABLE and self.ax is not None


def create_base_chart(
    title: str = "Chart",
    figsize: tuple[float, float] = (6, 4),
    parent: QWidget | None = None,
) -> BaseChartWidget:
    """Factory function to create a base chart widget.

    Args:
        title: Chart title
        figsize: Figure size as (width, height)
        parent: Parent widget

    Returns:
        BaseChartWidget instance
    """
    return BaseChartWidget(title=title, figsize=figsize, parent=parent)
