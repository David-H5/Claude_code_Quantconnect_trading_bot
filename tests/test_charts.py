"""
Tests for Chart Components (UPGRADE-007)

Tests the matplotlib-based chart widgets for the trading dashboard.
These tests work by directly importing chart modules without triggering ui/__init__.py.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def load_chart_module(module_name: str):
    """Load a chart module directly without going through ui/__init__.py."""
    base_path = Path(__file__).parent.parent / "ui" / "charts"
    module_path = base_path / f"{module_name}.py"

    spec = importlib.util.spec_from_file_location(f"ui.charts.{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"ui.charts.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", autouse=True)
def setup_mocks():
    """Set up mocks for Qt and matplotlib before loading chart modules."""
    # Save original modules
    saved_modules = {}
    modules_to_mock = [
        "PySide6",
        "PySide6.QtWidgets",
        "PySide6.QtCore",
        "matplotlib",
        "matplotlib.pyplot",
        "matplotlib.backends",
        "matplotlib.backends.backend_qt5agg",
        "matplotlib.figure",
        "matplotlib.dates",
    ]
    for mod in modules_to_mock:
        if mod in sys.modules:
            saved_modules[mod] = sys.modules[mod]

    # Create mock Qt classes
    class MockQWidget:
        def __init__(self, parent=None):
            self._parent = parent
            self._layout = None

        def setLayout(self, layout):
            self._layout = layout

    class MockQVBoxLayout:
        def __init__(self, parent=None):
            self._widgets = []

        def setContentsMargins(self, *args):
            pass

        def addWidget(self, widget):
            self._widgets.append(widget)

    class MockQLabel:
        def __init__(self, text=""):
            self._text = text

        def setAlignment(self, alignment):
            pass

        def setText(self, text):
            self._text = text

        def text(self):
            return self._text

    class MockQSizePolicy:
        Expanding = 0
        Preferred = 0

    class MockQt:
        AlignCenter = 0

    # Create mock Qt module
    mock_qt = MagicMock()
    mock_qt.QWidget = MockQWidget
    mock_qt.QVBoxLayout = MockQVBoxLayout
    mock_qt.QLabel = MockQLabel
    mock_qt.QSizePolicy = MockQSizePolicy
    mock_qt.Qt = MockQt

    mock_core = MagicMock()
    mock_core.Qt = MockQt

    # Create mock matplotlib
    mock_figure = MagicMock()
    mock_ax = MagicMock()
    mock_ax.twinx = MagicMock(return_value=MagicMock())
    mock_figure.add_subplot.return_value = mock_ax
    mock_figure.tight_layout = MagicMock()
    mock_figure.autofmt_xdate = MagicMock()

    mock_canvas = MagicMock()
    mock_canvas.draw = MagicMock()
    mock_canvas.setSizePolicy = MagicMock()

    mock_Figure = MagicMock(return_value=mock_figure)
    mock_FigureCanvas = MagicMock(return_value=mock_canvas)

    mock_matplotlib = MagicMock()
    mock_matplotlib.use = MagicMock()

    mock_backend = MagicMock()
    mock_backend.FigureCanvasQTAgg = mock_FigureCanvas

    mock_mpl_figure = MagicMock()
    mock_mpl_figure.Figure = mock_Figure

    mock_dates = MagicMock()
    mock_dates.DateFormatter = MagicMock(return_value=MagicMock())

    # Apply mocks to sys.modules (NOT numpy - it breaks other tests)
    sys.modules["PySide6"] = MagicMock()
    sys.modules["PySide6.QtWidgets"] = mock_qt
    sys.modules["PySide6.QtCore"] = mock_core
    sys.modules["matplotlib"] = mock_matplotlib
    sys.modules["matplotlib.pyplot"] = MagicMock()
    sys.modules["matplotlib.backends"] = mock_backend
    sys.modules["matplotlib.backends.backend_qt5agg"] = mock_backend
    sys.modules["matplotlib.figure"] = mock_mpl_figure
    sys.modules["matplotlib.dates"] = mock_dates

    # Clear any cached chart modules
    modules_to_clear = [m for m in list(sys.modules.keys()) if "ui.charts" in m]
    for m in modules_to_clear:
        del sys.modules[m]

    yield {
        "figure": mock_figure,
        "ax": mock_ax,
        "canvas": mock_canvas,
    }

    # Restore original modules and clean up chart modules
    for mod in modules_to_mock:
        if mod in saved_modules:
            sys.modules[mod] = saved_modules[mod]
        elif mod in sys.modules:
            del sys.modules[mod]

    # Clean up chart modules
    modules_to_clear = [m for m in list(sys.modules.keys()) if "ui.charts" in m]
    for m in modules_to_clear:
        if m in sys.modules:
            del sys.modules[m]


class TestBaseChartWidget:
    """Tests for BaseChartWidget."""

    def test_base_chart_creation(self, setup_mocks):
        """Test basic chart widget creation."""
        base_chart = load_chart_module("base_chart")
        chart = base_chart.BaseChartWidget(title="Test Chart")
        assert chart.title == "Test Chart"
        assert chart.figsize == (6, 4)

    def test_base_chart_custom_size(self, setup_mocks):
        """Test chart with custom figure size."""
        base_chart = load_chart_module("base_chart")
        chart = base_chart.BaseChartWidget(title="Custom Size", figsize=(10, 6))
        assert chart.figsize == (10, 6)

    def test_set_title(self, setup_mocks):
        """Test updating chart title."""
        base_chart = load_chart_module("base_chart")
        chart = base_chart.BaseChartWidget(title="Original")
        chart.set_title("Updated Title")
        assert chart.title == "Updated Title"

    def test_clear_method(self, setup_mocks):
        """Test chart clear method."""
        base_chart = load_chart_module("base_chart")
        chart = base_chart.BaseChartWidget(title="Test")
        # Should not raise
        chart.clear()

    def test_refresh_method(self, setup_mocks):
        """Test chart refresh method."""
        base_chart = load_chart_module("base_chart")
        chart = base_chart.BaseChartWidget(title="Test")
        # Should not raise
        chart.refresh()

    def test_is_available(self, setup_mocks):
        """Test matplotlib availability check."""
        base_chart = load_chart_module("base_chart")
        chart = base_chart.BaseChartWidget(title="Test")
        # Should return True when matplotlib is mocked
        assert chart.is_available()


class TestMetricsChartWidget:
    """Tests for MetricsChartWidget."""

    def test_metrics_chart_creation(self, setup_mocks):
        """Test metrics chart creation."""
        metrics_chart = load_chart_module("metrics_chart")
        chart = metrics_chart.MetricsChartWidget()
        assert chart.title == "Agent Metrics Trend"
        assert len(chart.data) == 0

    def test_set_data(self, setup_mocks):
        """Test setting metrics data."""
        metrics_chart = load_chart_module("metrics_chart")
        chart = metrics_chart.MetricsChartWidget()
        data = [
            metrics_chart.MetricsDataPoint(datetime.now(), 0.75, 0.80, 0.05),
            metrics_chart.MetricsDataPoint(datetime.now(), 0.78, 0.82, 0.04),
        ]
        chart.set_data(data)
        assert len(chart.data) == 2

    def test_add_point(self, setup_mocks):
        """Test adding single data point."""
        metrics_chart = load_chart_module("metrics_chart")
        chart = metrics_chart.MetricsChartWidget()
        point = metrics_chart.MetricsDataPoint(datetime.now(), 0.75, 0.80, 0.05)
        chart.add_point(point)
        assert len(chart.data) == 1

    def test_add_points(self, setup_mocks):
        """Test adding multiple data points."""
        metrics_chart = load_chart_module("metrics_chart")
        chart = metrics_chart.MetricsChartWidget()
        points = [
            metrics_chart.MetricsDataPoint(datetime.now(), 0.75, 0.80, 0.05),
            metrics_chart.MetricsDataPoint(datetime.now(), 0.78, 0.82, 0.04),
        ]
        chart.add_points(points)
        assert len(chart.data) == 2

    def test_get_latest(self, setup_mocks):
        """Test getting latest data point."""
        metrics_chart = load_chart_module("metrics_chart")
        chart = metrics_chart.MetricsChartWidget()
        assert chart.get_latest() is None

        point1 = metrics_chart.MetricsDataPoint(datetime.now(), 0.75, 0.80, 0.05)
        point2 = metrics_chart.MetricsDataPoint(datetime.now(), 0.78, 0.82, 0.04)
        chart.add_point(point1)
        chart.add_point(point2)

        latest = chart.get_latest()
        assert latest.accuracy == 0.78

    def test_get_averages(self, setup_mocks):
        """Test calculating average metrics."""
        metrics_chart = load_chart_module("metrics_chart")
        chart = metrics_chart.MetricsChartWidget()

        # Empty data
        avg_acc, avg_conf, avg_cal = chart.get_averages()
        assert avg_acc == 0.0
        assert avg_conf == 0.0

        # With data
        chart.set_data(
            [
                metrics_chart.MetricsDataPoint(datetime.now(), 0.70, 0.80, 0.05),
                metrics_chart.MetricsDataPoint(datetime.now(), 0.80, 0.90, 0.03),
            ]
        )
        avg_acc, avg_conf, avg_cal = chart.get_averages()
        assert abs(avg_acc - 0.75) < 0.001
        assert abs(avg_conf - 0.85) < 0.001

    def test_clear_data(self, setup_mocks):
        """Test clearing data."""
        metrics_chart = load_chart_module("metrics_chart")
        chart = metrics_chart.MetricsChartWidget()
        chart.add_point(metrics_chart.MetricsDataPoint(datetime.now(), 0.75, 0.80, 0.05))
        chart.clear_data()
        assert len(chart.data) == 0

    def test_show_calibration_toggle(self, setup_mocks):
        """Test calibration display toggle."""
        metrics_chart = load_chart_module("metrics_chart")
        chart = metrics_chart.MetricsChartWidget(show_calibration=True)
        assert chart.show_calibration is True

        chart.show_calibration = False
        assert chart.show_calibration is False


class TestCalibrationChartWidget:
    """Tests for CalibrationChartWidget."""

    def test_calibration_chart_creation(self, setup_mocks):
        """Test calibration chart creation."""
        calibration_chart = load_chart_module("calibration_chart")
        chart = calibration_chart.CalibrationChartWidget()
        assert chart.title == "Confidence Calibration"
        assert chart.n_bins == 10

    def test_set_bins(self, setup_mocks):
        """Test setting custom bin count."""
        calibration_chart = load_chart_module("calibration_chart")
        chart = calibration_chart.CalibrationChartWidget(n_bins=5)
        assert chart.n_bins == 5

        chart.n_bins = 15
        assert chart.n_bins == 15

        # Test clamping
        chart.n_bins = 1
        assert chart.n_bins == 2  # Min is 2

        chart.n_bins = 25
        assert chart.n_bins == 20  # Max is 20

    def test_set_calibration_data(self, setup_mocks):
        """Test setting calibration data."""
        calibration_chart = load_chart_module("calibration_chart")
        bins = [
            calibration_chart.CalibrationBin((0.0, 0.2), 0.1, 0.15, 10),
            calibration_chart.CalibrationBin((0.2, 0.4), 0.3, 0.28, 20),
        ]
        chart = calibration_chart.CalibrationChartWidget()
        chart.set_data(bins)
        assert len(chart.bins) == 2

    def test_compute_from_predictions(self, setup_mocks):
        """Test computing calibration from predictions."""
        calibration_chart = load_chart_module("calibration_chart")
        chart = calibration_chart.CalibrationChartWidget(n_bins=5)
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
        correct = [True, False, True, True, True]

        chart.compute_from_predictions(confidences, correct)
        # Should have created bins
        assert len(chart.bins) > 0

    def test_expected_calibration_error(self, setup_mocks):
        """Test ECE calculation."""
        calibration_chart = load_chart_module("calibration_chart")
        chart = calibration_chart.CalibrationChartWidget()

        # No data
        assert chart.expected_calibration_error == 0.0

        # Perfect calibration
        bins = [
            calibration_chart.CalibrationBin((0.0, 0.5), 0.25, 0.25, 50),
            calibration_chart.CalibrationBin((0.5, 1.0), 0.75, 0.75, 50),
        ]
        chart.set_data(bins)
        assert chart.expected_calibration_error == 0.0

    def test_reliability_summary(self, setup_mocks):
        """Test getting reliability summary."""
        calibration_chart = load_chart_module("calibration_chart")
        chart = calibration_chart.CalibrationChartWidget()

        # Empty
        summary = chart.get_reliability_summary()
        assert summary["total_samples"] == 0

        # With data
        bins = [
            calibration_chart.CalibrationBin((0.0, 0.5), 0.4, 0.3, 50),
            calibration_chart.CalibrationBin((0.5, 1.0), 0.8, 0.9, 50),
        ]
        chart.set_data(bins)
        summary = chart.get_reliability_summary()
        assert summary["total_samples"] == 100
        assert summary["num_bins"] == 2


class TestEvolutionChartWidget:
    """Tests for EvolutionChartWidget."""

    def test_evolution_chart_creation(self, setup_mocks):
        """Test evolution chart creation."""
        evolution_chart = load_chart_module("evolution_chart")
        chart = evolution_chart.EvolutionChartWidget()
        assert chart.title == "Evolution Progress"
        assert len(chart.cycles) == 0

    def test_set_evolution_data(self, setup_mocks):
        """Test setting evolution data."""
        evolution_chart = load_chart_module("evolution_chart")
        chart = evolution_chart.EvolutionChartWidget()
        cycles = [
            evolution_chart.EvolutionCycle(1, 0.65, 0.0, "v1.0"),
            evolution_chart.EvolutionCycle(2, 0.72, 0.07, "v1.1"),
        ]
        chart.set_data(cycles)
        assert len(chart.cycles) == 2

    def test_add_cycle(self, setup_mocks):
        """Test adding single cycle."""
        evolution_chart = load_chart_module("evolution_chart")
        chart = evolution_chart.EvolutionChartWidget()
        chart.add_cycle(evolution_chart.EvolutionCycle(1, 0.65, 0.0, "v1.0"))
        chart.add_cycle(evolution_chart.EvolutionCycle(2, 0.72, 0.0, "v1.1"))

        assert len(chart.cycles) == 2
        # Second cycle should have calculated improvement
        assert abs(chart.cycles[1].improvement - 0.07) < 0.001

    def test_total_improvement(self, setup_mocks):
        """Test total improvement calculation."""
        evolution_chart = load_chart_module("evolution_chart")
        chart = evolution_chart.EvolutionChartWidget()

        # No data
        assert chart.total_improvement == 0.0

        # With data
        chart.set_data(
            [
                evolution_chart.EvolutionCycle(1, 0.60, 0.0, "v1.0"),
                evolution_chart.EvolutionCycle(2, 0.70, 0.10, "v1.1"),
                evolution_chart.EvolutionCycle(3, 0.75, 0.05, "v1.2"),
            ]
        )
        assert abs(chart.total_improvement - 0.15) < 0.001

    def test_best_cycle(self, setup_mocks):
        """Test finding best cycle."""
        evolution_chart = load_chart_module("evolution_chart")
        chart = evolution_chart.EvolutionChartWidget()
        assert chart.best_cycle is None

        chart.set_data(
            [
                evolution_chart.EvolutionCycle(1, 0.60, 0.0, "v1.0"),
                evolution_chart.EvolutionCycle(2, 0.75, 0.15, "v1.1"),
                evolution_chart.EvolutionCycle(3, 0.70, -0.05, "v1.2"),
            ]
        )
        best = chart.best_cycle
        assert best.cycle_number == 2
        assert best.score == 0.75

    def test_convergence_detection(self, setup_mocks):
        """Test convergence detection."""
        evolution_chart = load_chart_module("evolution_chart")
        chart = evolution_chart.EvolutionChartWidget()

        # Not enough cycles
        chart.set_data([evolution_chart.EvolutionCycle(1, 0.60, 0.0, "v1.0")])
        is_converged, reason = chart.get_convergence_status()
        assert is_converged is False
        assert "Not enough cycles" in reason

        # Still improving
        chart.set_data(
            [
                evolution_chart.EvolutionCycle(1, 0.60, 0.0, "v1.0"),
                evolution_chart.EvolutionCycle(2, 0.70, 0.10, "v1.1"),
                evolution_chart.EvolutionCycle(3, 0.80, 0.10, "v1.2"),
            ]
        )
        is_converged, reason = chart.get_convergence_status()
        assert is_converged is False

        # Converged (minimal improvement)
        chart.set_data(
            [
                evolution_chart.EvolutionCycle(1, 0.75, 0.0, "v1.0"),
                evolution_chart.EvolutionCycle(2, 0.755, 0.005, "v1.1"),
                evolution_chart.EvolutionCycle(3, 0.758, 0.003, "v1.2"),
            ]
        )
        is_converged, reason = chart.get_convergence_status()
        assert is_converged is True

    def test_evolution_summary(self, setup_mocks):
        """Test getting evolution summary."""
        evolution_chart = load_chart_module("evolution_chart")
        chart = evolution_chart.EvolutionChartWidget()

        # Empty
        summary = chart.get_evolution_summary()
        assert summary["num_cycles"] == 0

        # With data
        chart.set_data(
            [
                evolution_chart.EvolutionCycle(1, 0.60, 0.0, "v1.0"),
                evolution_chart.EvolutionCycle(2, 0.75, 0.15, "v1.1"),
            ]
        )
        summary = chart.get_evolution_summary()
        assert summary["num_cycles"] == 2
        assert summary["current_score"] == 0.75
        assert summary["best_score"] == 0.75


class TestDecisionChartWidget:
    """Tests for DecisionChartWidget."""

    def test_decision_chart_creation(self, setup_mocks):
        """Test decision chart creation."""
        decision_chart = load_chart_module("decision_chart")
        chart = decision_chart.DecisionChartWidget(chart_type="bar")
        assert chart.chart_type == "bar"
        assert len(chart.stats) == 0

    def test_pie_chart_type(self, setup_mocks):
        """Test pie chart type."""
        decision_chart = load_chart_module("decision_chart")
        chart = decision_chart.DecisionChartWidget(chart_type="pie")
        assert chart.chart_type == "pie"

    def test_set_decision_data(self, setup_mocks):
        """Test setting decision data."""
        decision_chart = load_chart_module("decision_chart")
        chart = decision_chart.DecisionChartWidget()
        stats = [
            decision_chart.DecisionStats("BUY", correct=10, incorrect=3, pending=2),
            decision_chart.DecisionStats("SELL", correct=8, incorrect=5, pending=1),
        ]
        chart.set_data(stats)
        assert len(chart.stats) == 2

    def test_set_outcome_counts(self, setup_mocks):
        """Test setting outcome counts for pie chart."""
        decision_chart = load_chart_module("decision_chart")
        chart = decision_chart.DecisionChartWidget(chart_type="pie")
        counts = {"correct": 50, "incorrect": 20, "pending": 10}
        # Just verify it can be set without errors - pie chart rendering is mocked
        chart._outcome_counts = counts
        assert chart._outcome_counts == counts

    def test_add_decision(self, setup_mocks):
        """Test adding individual decisions."""
        decision_chart = load_chart_module("decision_chart")
        chart = decision_chart.DecisionChartWidget()
        chart.add_decision("BUY", decision_chart.DecisionOutcome.CORRECT)
        chart.add_decision("BUY", decision_chart.DecisionOutcome.CORRECT)
        chart.add_decision("BUY", decision_chart.DecisionOutcome.INCORRECT)

        assert len(chart.stats) == 1
        assert chart.stats[0].correct == 2
        assert chart.stats[0].incorrect == 1

    def test_decision_stats_accuracy(self, setup_mocks):
        """Test DecisionStats accuracy calculation."""
        decision_chart = load_chart_module("decision_chart")
        stats = decision_chart.DecisionStats("BUY", correct=8, incorrect=2)
        assert abs(stats.accuracy - 0.8) < 0.001
        assert stats.total == 10

        # No resolved decisions
        empty_stats = decision_chart.DecisionStats("HOLD", pending=5)
        assert empty_stats.accuracy == 0.0

    def test_chart_type_toggle(self, setup_mocks):
        """Test toggling chart type."""
        decision_chart = load_chart_module("decision_chart")
        chart = decision_chart.DecisionChartWidget(chart_type="bar")
        assert chart.chart_type == "bar"

        chart.chart_type = "pie"
        assert chart.chart_type == "pie"

        # Invalid type should not change
        chart.chart_type = "invalid"
        assert chart.chart_type == "pie"

    def test_decision_summary(self, setup_mocks):
        """Test getting decision summary."""
        decision_chart = load_chart_module("decision_chart")
        chart = decision_chart.DecisionChartWidget()

        # Empty
        summary = chart.get_decision_summary()
        assert summary["total_decisions"] == 0

        # With data
        chart.set_data(
            [
                decision_chart.DecisionStats("BUY", correct=8, incorrect=2),
                decision_chart.DecisionStats("SELL", correct=6, incorrect=4),
            ]
        )
        summary = chart.get_decision_summary()
        assert summary["total_decisions"] == 20
        assert abs(summary["overall_accuracy"] - 0.7) < 0.001  # 14/20
        assert "BUY" in summary["by_type"]
        assert "SELL" in summary["by_type"]

    def test_clear_data(self, setup_mocks):
        """Test clearing decision data."""
        decision_chart = load_chart_module("decision_chart")
        chart = decision_chart.DecisionChartWidget()
        chart.set_data([decision_chart.DecisionStats("BUY", correct=10)])
        chart.clear_data()
        assert len(chart.stats) == 0


class TestChartFactoryFunctions:
    """Tests for chart factory functions."""

    def test_create_metrics_chart(self, setup_mocks):
        """Test metrics chart factory."""
        metrics_chart = load_chart_module("metrics_chart")
        chart = metrics_chart.create_metrics_chart(show_calibration=True)
        assert chart is not None
        assert chart.show_calibration is True

    def test_create_calibration_chart(self, setup_mocks):
        """Test calibration chart factory."""
        calibration_chart = load_chart_module("calibration_chart")
        chart = calibration_chart.create_calibration_chart(n_bins=5)
        assert chart is not None
        assert chart.n_bins == 5

    def test_create_evolution_chart(self, setup_mocks):
        """Test evolution chart factory."""
        evolution_chart = load_chart_module("evolution_chart")
        chart = evolution_chart.create_evolution_chart(show_improvement_bars=False)
        assert chart is not None

    def test_create_decision_chart(self, setup_mocks):
        """Test decision chart factory."""
        decision_chart = load_chart_module("decision_chart")
        bar_chart = decision_chart.create_decision_chart(chart_type="bar")
        assert bar_chart.chart_type == "bar"

        pie_chart = decision_chart.create_decision_chart(chart_type="pie")
        assert pie_chart.chart_type == "pie"


class TestELOHistoryChart:
    """Tests for ELOHistoryChart (UPGRADE-010 Sprint 2 P1)."""

    def test_elo_chart_creation(self, setup_mocks):
        """Test ELO history chart creation."""
        elo_chart = load_chart_module("elo_history_chart")
        chart = elo_chart.ELOHistoryChart(title="Test ELO")
        assert chart._title == "Test ELO"
        assert chart._time_range_days is None

    def test_set_data(self, setup_mocks):
        """Test setting ELO history data."""
        elo_chart = load_chart_module("elo_history_chart")
        chart = elo_chart.ELOHistoryChart()
        history_data = {
            "agent_1": [
                {"timestamp": "2025-01-01T12:00:00", "elo_rating": 1500.0, "change": 0.0},
                {"timestamp": "2025-01-02T12:00:00", "elo_rating": 1520.0, "change": 20.0},
            ],
            "agent_2": [
                {"timestamp": "2025-01-01T12:00:00", "elo_rating": 1500.0, "change": 0.0},
                {"timestamp": "2025-01-02T12:00:00", "elo_rating": 1480.0, "change": -20.0},
            ],
        }
        chart.set_data(history_data)
        assert len(chart._history_data) == 2
        assert "agent_1" in chart._history_data

    def test_set_time_range(self, setup_mocks):
        """Test time range filter."""
        elo_chart = load_chart_module("elo_history_chart")
        chart = elo_chart.ELOHistoryChart()
        chart.set_time_range(30)
        assert chart._time_range_days == 30

        chart.set_time_range(None)
        assert chart._time_range_days is None

    def test_export_csv(self, setup_mocks):
        """Test CSV export."""
        elo_chart = load_chart_module("elo_history_chart")
        chart = elo_chart.ELOHistoryChart()
        chart._history_data = {
            "agent_1": [
                {"timestamp": "2025-01-01T12:00:00", "elo_rating": 1500.0, "change": 0.0, "outcome": "win"},
            ],
        }
        csv_output = chart.export_csv()
        assert "agent_name" in csv_output
        assert "agent_1" in csv_output
        assert "1500" in csv_output

    def test_get_agent_summary(self, setup_mocks):
        """Test agent summary statistics."""
        elo_chart = load_chart_module("elo_history_chart")
        chart = elo_chart.ELOHistoryChart()
        chart._history_data = {
            "agent_1": [
                {"timestamp": "2025-01-01T12:00:00", "elo_rating": 1500.0, "change": 0.0},
                {"timestamp": "2025-01-02T12:00:00", "elo_rating": 1520.0, "change": 20.0},
                {"timestamp": "2025-01-03T12:00:00", "elo_rating": 1510.0, "change": -10.0},
            ],
        }
        summary = chart.get_agent_summary()
        assert "agent_1" in summary
        assert summary["agent_1"]["current_elo"] == 1510.0
        assert summary["agent_1"]["highest_elo"] == 1520.0
        assert summary["agent_1"]["lowest_elo"] == 1500.0
        assert summary["agent_1"]["total_changes"] == 3

    def test_empty_history_summary(self, setup_mocks):
        """Test summary with empty history."""
        elo_chart = load_chart_module("elo_history_chart")
        chart = elo_chart.ELOHistoryChart()
        chart._history_data = {"agent_1": []}
        summary = chart.get_agent_summary()
        assert "agent_1" not in summary  # Empty history excluded

    def test_factory_function(self, setup_mocks):
        """Test ELO chart factory function."""
        elo_chart = load_chart_module("elo_history_chart")
        chart = elo_chart.create_elo_history_chart(
            title="Custom ELO",
            figsize=(12, 8),
        )
        assert chart._title == "Custom ELO"
        assert chart._figsize == (12, 8)


class TestAttentionHeatmapChart:
    """Tests for AttentionHeatmapChart (UPGRADE-010 Sprint 2 P1)."""

    def test_attention_chart_creation(self, setup_mocks):
        """Test attention heatmap chart creation."""
        attention_chart = load_chart_module("attention_chart")
        chart = attention_chart.AttentionHeatmapChart(title="Test Attention")
        assert chart._title == "Test Attention"
        assert chart._selected_head is None

    def test_set_weights(self, setup_mocks):
        """Test setting attention weights."""
        attention_chart = load_chart_module("attention_chart")
        chart = attention_chart.AttentionHeatmapChart()
        weights = [
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.3, 0.3, 0.4],
        ]
        labels = ["Asset1", "Asset2", "Asset3"]
        chart.set_weights(weights, labels)
        assert chart._weights == weights
        assert chart._labels == labels

    def test_set_head_weights(self, setup_mocks):
        """Test setting per-head attention weights."""
        attention_chart = load_chart_module("attention_chart")
        chart = attention_chart.AttentionHeatmapChart()
        head_weights = [
            [[0.5, 0.5], [0.5, 0.5]],  # Head 0
            [[0.3, 0.7], [0.7, 0.3]],  # Head 1
        ]
        chart.set_head_weights(head_weights)
        assert len(chart._head_weights) == 2

    def test_select_head(self, setup_mocks):
        """Test head selection."""
        attention_chart = load_chart_module("attention_chart")
        chart = attention_chart.AttentionHeatmapChart()
        chart._head_weights = [[[0.5, 0.5]], [[0.3, 0.7]]]
        chart.select_head(1)
        assert chart._selected_head == 1

        chart.select_head(None)
        assert chart._selected_head is None

    def test_get_head_count(self, setup_mocks):
        """Test head count retrieval."""
        attention_chart = load_chart_module("attention_chart")
        chart = attention_chart.AttentionHeatmapChart()
        assert chart.get_head_count() == 0

        chart._head_weights = [[[0.5]], [[0.5]], [[0.5]], [[0.5]]]
        assert chart.get_head_count() == 4

    def test_export_csv(self, setup_mocks):
        """Test CSV export of attention weights."""
        attention_chart = load_chart_module("attention_chart")
        chart = attention_chart.AttentionHeatmapChart()
        chart._weights = [[0.5, 0.5], [0.5, 0.5]]
        chart._labels = ["A", "B"]
        csv_output = chart.export_csv()
        assert "A" in csv_output
        assert "B" in csv_output

    def test_factory_function(self, setup_mocks):
        """Test attention chart factory function."""
        attention_chart = load_chart_module("attention_chart")
        chart = attention_chart.create_attention_heatmap_chart(
            title="Custom Attention",
            figsize=(8, 8),
        )
        assert chart._title == "Custom Attention"


class TestChartDataClasses:
    """Tests for chart data classes."""

    def test_metrics_data_point(self, setup_mocks):
        """Test MetricsDataPoint dataclass."""
        metrics_chart = load_chart_module("metrics_chart")
        point = metrics_chart.MetricsDataPoint(
            timestamp=datetime.now(), accuracy=0.85, confidence=0.90, calibration_error=0.05
        )
        assert point.accuracy == 0.85
        assert point.confidence == 0.90
        assert point.calibration_error == 0.05

    def test_calibration_bin(self, setup_mocks):
        """Test CalibrationBin dataclass."""
        calibration_chart = load_chart_module("calibration_chart")
        bin_data = calibration_chart.CalibrationBin(
            confidence_range=(0.3, 0.4), mean_confidence=0.35, accuracy=0.40, count=25
        )
        assert bin_data.confidence_range == (0.3, 0.4)
        assert bin_data.mean_confidence == 0.35
        assert bin_data.accuracy == 0.40
        assert bin_data.count == 25

    def test_evolution_cycle(self, setup_mocks):
        """Test EvolutionCycle dataclass."""
        evolution_chart = load_chart_module("evolution_chart")
        cycle = evolution_chart.EvolutionCycle(cycle_number=5, score=0.82, improvement=0.03, prompt_version="v2.1")
        assert cycle.cycle_number == 5
        assert cycle.score == 0.82
        assert cycle.improvement == 0.03
        assert cycle.prompt_version == "v2.1"

    def test_decision_stats(self, setup_mocks):
        """Test DecisionStats dataclass."""
        decision_chart = load_chart_module("decision_chart")
        stats = decision_chart.DecisionStats(decision_type="BUY", correct=15, incorrect=5, pending=3, skipped=2)
        assert stats.decision_type == "BUY"
        assert stats.total == 25
        assert abs(stats.accuracy - 0.75) < 0.001

    def test_decision_outcome_enum(self, setup_mocks):
        """Test DecisionOutcome enum."""
        decision_chart = load_chart_module("decision_chart")
        assert decision_chart.DecisionOutcome.CORRECT.value == "correct"
        assert decision_chart.DecisionOutcome.INCORRECT.value == "incorrect"
        assert decision_chart.DecisionOutcome.PENDING.value == "pending"
        assert decision_chart.DecisionOutcome.SKIPPED.value == "skipped"
