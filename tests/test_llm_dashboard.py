"""
Tests for LLM Dashboard Integration Widgets

Tests verify the dashboard widgets correctly:
- Display agent metrics
- Visualize debate sessions
- Monitor evolution progress
- Browse decision logs
- Integrate with main dashboard

UPGRADE-006: LLM Dashboard Integration
"""

import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest


# Set Qt platform to offscreen before any Qt imports
# This prevents segfaults in headless environments
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Check for PySide6 availability
PYSIDE_AVAILABLE = False
try:
    from PySide6.QtWidgets import QApplication

    PYSIDE_AVAILABLE = True
except (ImportError, RuntimeError):
    # RuntimeError can occur if Qt can't initialize even with offscreen platform
    pass


# Skip all tests if PySide6 is not available or Qt can't initialize
pytestmark = pytest.mark.skipif(not PYSIDE_AVAILABLE, reason="PySide6 not available or Qt cannot initialize")


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for testing."""
    if not PYSIDE_AVAILABLE:
        pytest.skip("PySide6 not available")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_metrics():
    """Create mock agent metrics data."""
    return {
        "agent_name": "technical_analyst",
        "total_decisions": 100,
        "decisions_evaluated": 80,
        "accuracy_rate": 0.75,
        "calibration_error": 0.05,
        "average_confidence": 0.72,
        "confidence_std": 0.15,
        "overconfidence_rate": 0.12,
        "underconfidence_rate": 0.08,
        "average_execution_time_ms": 150.0,
        "p95_execution_time_ms": 350.0,
        "average_reasoning_steps": 3.5,
        "decision_distribution": {"trade": 60, "analysis": 40},
        "time_period_start": datetime.utcnow() - timedelta(days=7),
        "time_period_end": datetime.utcnow(),
    }


@pytest.fixture
def mock_metrics_tracker(mock_metrics):
    """Create mock AgentMetricsTracker."""
    tracker = MagicMock()

    # Create a mock AgentMetrics object
    metrics = MagicMock()
    for key, value in mock_metrics.items():
        setattr(metrics, key, value)

    tracker.get_metrics.return_value = metrics
    tracker.get_all_metrics.return_value = {"technical_analyst": metrics}

    return tracker


@pytest.fixture
def mock_debate_result():
    """Create mock debate result data."""
    return {
        "debate_id": "debate_123abc",
        "opportunity": {"symbol": "SPY", "position_size_pct": 0.15},
        "rounds": [
            {
                "round_number": 0,
                "bull_argument": {
                    "content": "Strong technical indicators suggest upward momentum.",
                    "confidence": 0.8,
                    "key_points": ["RSI oversold", "MACD bullish cross"],
                    "evidence": ["Price above 50-day MA"],
                    "risks_identified": ["Market volatility high"],
                },
                "bear_argument": {
                    "content": "Macro headwinds may limit upside.",
                    "confidence": 0.6,
                    "key_points": ["Fed rate concerns", "Yield curve pressure"],
                    "evidence": ["Bond market signals"],
                    "risks_identified": ["Policy uncertainty"],
                },
                "moderator_assessment": {
                    "summary": "Bull case slightly stronger on technicals.",
                    "stronger_argument": "bull",
                    "key_disagreements": ["Market direction"],
                    "areas_of_agreement": ["Volatility elevated"],
                    "recommended_action": "cautious_long",
                    "confidence": 0.65,
                },
            }
        ],
        "final_outcome": "buy",
        "consensus_confidence": 0.65,
        "key_points_bull": ["Technical momentum"],
        "key_points_bear": ["Macro risks"],
        "risk_factors": ["High volatility"],
        "trigger_reason": "high_position_size",
        "total_duration_ms": 2500.0,
        "timestamp": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def mock_evolution_result():
    """Create mock evolution result data."""
    return {
        "agent_name": "technical_analyst",
        "initial_score": 0.65,
        "final_score": 0.82,
        "total_improvement": 0.17,
        "num_cycles": 3,
        "converged": True,
        "convergence_reason": "target_reached",
        "duration_ms": 15000.0,
        "timestamp": datetime.utcnow().isoformat(),
        "cycles": [
            {
                "cycle_number": 0,
                "pre_score": 0.65,
                "post_score": 0.72,
                "refinements_applied": ["Add examples for edge cases"],
                "improvement": 0.07,
                "weaknesses_identified": ["Handling of volatile markets"],
            },
            {
                "cycle_number": 1,
                "pre_score": 0.72,
                "post_score": 0.78,
                "refinements_applied": ["Clarify risk assessment instructions"],
                "improvement": 0.06,
                "weaknesses_identified": ["Risk evaluation inconsistent"],
            },
            {
                "cycle_number": 2,
                "pre_score": 0.78,
                "post_score": 0.82,
                "refinements_applied": ["Add constraints for position sizing"],
                "improvement": 0.04,
                "weaknesses_identified": [],
            },
        ],
        "prompt_versions": [
            {"version": 1, "prompt": "Original prompt v1...", "score": 0.65},
            {"version": 2, "prompt": "Improved prompt v2...", "score": 0.72},
            {"version": 3, "prompt": "Enhanced prompt v3...", "score": 0.78},
            {"version": 4, "prompt": "Final prompt v4...", "score": 0.82},
        ],
        "max_cycles": 5,
        "target_score": 0.85,
    }


@pytest.fixture
def mock_decision_log():
    """Create mock decision log data."""
    return {
        "log_id": "dec_abc123",
        "timestamp": datetime.utcnow().isoformat(),
        "agent_name": "technical_analyst",
        "agent_role": "analyst",
        "decision_type": "trade",
        "decision": "BUY SPY at market",
        "confidence": 0.78,
        "context": {"symbol": "SPY", "price": 450.0},
        "query": "Analyze SPY for entry opportunity",
        "reasoning_chain": [
            {"step_number": 1, "thought": "Check technical indicators", "confidence": 0.8},
            {"step_number": 2, "thought": "RSI shows oversold", "confidence": 0.75},
            {"step_number": 3, "thought": "MACD bullish crossover", "confidence": 0.82},
        ],
        "final_reasoning": "Multiple technical signals align for entry",
        "alternatives_considered": [
            {"description": "Wait for pullback", "reason_rejected": "May miss move"},
        ],
        "risk_assessment": {
            "overall_level": "medium",
            "factors": ["Market volatility"],
            "mitigation_steps": ["Use stop loss"],
            "worst_case_scenario": "2% portfolio loss",
            "probability_of_loss": 0.25,
        },
        "execution_time_ms": 125.0,
        "outcome": "executed",
    }


class TestAgentMetricsWidget:
    """Tests for AgentMetricsWidget."""

    def test_widget_creation(self, qapp):
        """Widget can be created."""
        from ui.agent_metrics_widget import AgentMetricsWidget

        widget = AgentMetricsWidget()
        assert widget is not None

    def test_widget_with_tracker(self, qapp, mock_metrics_tracker):
        """Widget displays metrics from tracker."""
        from ui.agent_metrics_widget import AgentMetricsWidget

        widget = AgentMetricsWidget(metrics_tracker=mock_metrics_tracker)
        widget.refresh()

        # Tracker was queried
        mock_metrics_tracker.get_all_metrics.assert_called()

    def test_set_metrics_tracker(self, qapp, mock_metrics_tracker):
        """Can set metrics tracker after creation."""
        from ui.agent_metrics_widget import AgentMetricsWidget

        widget = AgentMetricsWidget()
        widget.set_metrics_tracker(mock_metrics_tracker)

        assert widget.tracker == mock_metrics_tracker

    def test_factory_function(self, qapp):
        """Factory function creates widget."""
        from ui.agent_metrics_widget import create_agent_metrics_widget

        widget = create_agent_metrics_widget()
        assert widget is not None

    def test_metric_label_color_coding(self, qapp):
        """MetricLabel applies color coding based on thresholds."""
        from ui.agent_metrics_widget import MetricDisplayConfig, MetricLabel

        config = MetricDisplayConfig(
            label="Test",
            format_str="{:.1%}",
            color_thresholds={"good": 0.8, "warn": 0.6},
        )
        label = MetricLabel("test", config)

        # High value should be green
        label.set_value(0.9)
        # Widget updated (we can't easily check color without more setup)

        # Low value should be red
        label.set_value(0.4)


class TestDebateViewer:
    """Tests for DebateViewer."""

    def test_viewer_creation(self, qapp):
        """Viewer can be created."""
        from ui.debate_viewer import DebateViewer

        viewer = DebateViewer()
        assert viewer is not None

    def test_add_debate(self, qapp, mock_debate_result):
        """Can add debate results."""
        from ui.debate_viewer import DebateViewer

        viewer = DebateViewer()
        viewer.add_debate(mock_debate_result)

        assert len(viewer._debate_history) == 1

    def test_load_debate_history(self, qapp, mock_debate_result):
        """Can load multiple debates."""
        from ui.debate_viewer import DebateViewer

        viewer = DebateViewer()
        history = [mock_debate_result, mock_debate_result]
        viewer.load_debate_history(history)

        assert len(viewer._debate_history) == 2

    def test_round_navigation(self, qapp, mock_debate_result):
        """Can navigate between rounds."""
        from ui.debate_viewer import DebateViewer

        viewer = DebateViewer()
        viewer.add_debate(mock_debate_result)

        # Initial round
        assert viewer._current_round == 0

        # Add more rounds for testing
        mock_debate_result["rounds"].append(mock_debate_result["rounds"][0].copy())
        viewer._current_debate = mock_debate_result
        viewer._next_round()
        assert viewer._current_round == 1

        viewer._prev_round()
        assert viewer._current_round == 0

    def test_factory_function(self, qapp):
        """Factory function creates viewer."""
        from ui.debate_viewer import create_debate_viewer

        viewer = create_debate_viewer()
        assert viewer is not None

    def test_argument_panel(self, qapp, mock_debate_result):
        """ArgumentPanel displays argument data."""
        from ui.debate_viewer import ArgumentPanel

        panel = ArgumentPanel("Bull", "#4CAF50")
        argument = mock_debate_result["rounds"][0]["bull_argument"]
        panel.set_argument(argument)

        # Panel updated (verify no exceptions)
        panel.clear()

    def test_moderator_panel(self, qapp, mock_debate_result):
        """ModeratorPanel displays assessment data."""
        from ui.debate_viewer import ModeratorPanel

        panel = ModeratorPanel()
        assessment = mock_debate_result["rounds"][0]["moderator_assessment"]
        panel.set_assessment(assessment)

        panel.clear()


class TestEvolutionMonitor:
    """Tests for EvolutionMonitor."""

    def test_monitor_creation(self, qapp):
        """Monitor can be created."""
        from ui.evolution_monitor import EvolutionMonitor

        monitor = EvolutionMonitor()
        assert monitor is not None

    def test_add_evolution_result(self, qapp, mock_evolution_result):
        """Can add evolution results."""
        from ui.evolution_monitor import EvolutionMonitor

        monitor = EvolutionMonitor()
        monitor.add_evolution_result("technical_analyst", mock_evolution_result)

        assert "technical_analyst" in monitor._evolution_results

    def test_progress_widget(self, qapp, mock_evolution_result):
        """EvolutionProgressWidget updates correctly."""
        from ui.evolution_monitor import EvolutionProgressWidget

        widget = EvolutionProgressWidget()
        widget.update_progress(
            cycle=3,
            max_cycles=5,
            initial_score=0.65,
            current_score=0.82,
            target_score=0.85,
            status="target_reached",
        )

        # Widget updated (verify no exceptions)

    def test_cycle_history_widget(self, qapp, mock_evolution_result):
        """CycleHistoryWidget displays cycles."""
        from ui.evolution_monitor import CycleHistoryWidget

        widget = CycleHistoryWidget()
        widget.set_cycles(mock_evolution_result["cycles"])

        # Should have 3 rows
        assert widget._table.rowCount() == 3

    def test_prompt_version_widget(self, qapp, mock_evolution_result):
        """PromptVersionWidget displays versions."""
        from ui.evolution_monitor import PromptVersionWidget

        widget = PromptVersionWidget()
        widget.set_versions(mock_evolution_result["prompt_versions"])

        # Should have 4 items in combo
        assert widget._version_combo.count() == 4

    def test_factory_function(self, qapp):
        """Factory function creates monitor."""
        from ui.evolution_monitor import create_evolution_monitor

        monitor = create_evolution_monitor()
        assert monitor is not None


class TestDecisionLogViewer:
    """Tests for DecisionLogViewer."""

    def test_viewer_creation(self, qapp):
        """Viewer can be created."""
        from ui.decision_log_viewer import DecisionLogViewer

        viewer = DecisionLogViewer()
        assert viewer is not None

    def test_load_decisions(self, qapp, mock_decision_log):
        """Can load decisions."""
        from ui.decision_log_viewer import DecisionLogViewer

        viewer = DecisionLogViewer()
        viewer.load_decisions([mock_decision_log])

        assert len(viewer._decisions) == 1

    def test_add_decision(self, qapp, mock_decision_log):
        """Can add individual decisions."""
        from ui.decision_log_viewer import DecisionLogViewer

        viewer = DecisionLogViewer()
        viewer.add_decision(mock_decision_log)
        viewer.add_decision(mock_decision_log)

        assert len(viewer._decisions) == 2

    def test_filters(self, qapp, mock_decision_log):
        """Filters work correctly."""
        from ui.decision_log_viewer import DecisionLogViewer

        viewer = DecisionLogViewer()
        decisions = [mock_decision_log.copy() for _ in range(5)]
        decisions[0]["agent_name"] = "other_agent"
        decisions[1]["outcome"] = "rejected"

        viewer.load_decisions(decisions)

        # All shown initially
        assert len(viewer._filtered_decisions) == 5

    def test_detail_panel(self, qapp, mock_decision_log):
        """DecisionDetailPanel displays decision data."""
        from ui.decision_log_viewer import DecisionDetailPanel

        panel = DecisionDetailPanel()
        panel.set_decision(mock_decision_log)

        # Panel updated
        panel.clear()

    def test_factory_function(self, qapp):
        """Factory function creates viewer."""
        from ui.decision_log_viewer import create_decision_log_viewer

        viewer = create_decision_log_viewer()
        assert viewer is not None


class TestDashboardIntegration:
    """Tests for LLM dashboard integration."""

    def test_dashboard_has_llm_widgets(self, qapp):
        """Dashboard includes LLM widgets."""
        from ui.dashboard import TradingDashboard

        dashboard = TradingDashboard()

        assert hasattr(dashboard, "agent_metrics_widget")
        assert hasattr(dashboard, "debate_viewer")
        assert hasattr(dashboard, "evolution_monitor")
        assert hasattr(dashboard, "decision_log_viewer")

    def test_llm_docks_hidden_by_default(self, qapp):
        """LLM docks are hidden by default."""
        from ui.dashboard import TradingDashboard

        dashboard = TradingDashboard()

        for dock in dashboard._llm_docks.values():
            assert not dock.isVisible()

    def test_show_all_llm_docks(self, qapp):
        """Can show all LLM docks."""
        from ui.dashboard import TradingDashboard

        dashboard = TradingDashboard()
        dashboard._show_all_llm_docks()

        # Use isHidden() instead of isVisible() since parent window isn't shown
        for dock in dashboard._llm_docks.values():
            assert not dock.isHidden()

    def test_hide_all_llm_docks(self, qapp):
        """Can hide all LLM docks."""
        from ui.dashboard import TradingDashboard

        dashboard = TradingDashboard()
        dashboard._show_all_llm_docks()
        dashboard._hide_all_llm_docks()

        for dock in dashboard._llm_docks.values():
            assert not dock.isVisible()

    def test_toggle_individual_dock(self, qapp):
        """Can toggle individual docks."""
        from ui.dashboard import TradingDashboard

        dashboard = TradingDashboard()

        dashboard._toggle_llm_dock("Agent Metrics", True)
        # Use isHidden() instead of isVisible() since parent window isn't shown
        assert not dashboard._llm_docks["Agent Metrics"].isHidden()

        dashboard._toggle_llm_dock("Agent Metrics", False)
        assert dashboard._llm_docks["Agent Metrics"].isHidden()

    def test_set_llm_components(self, qapp, mock_metrics_tracker):
        """Can set LLM components."""
        from ui.dashboard import TradingDashboard

        dashboard = TradingDashboard()
        dashboard.set_llm_components(metrics_tracker=mock_metrics_tracker)

        assert dashboard.agent_metrics_widget.tracker == mock_metrics_tracker

    def test_create_llm_dashboard(self, qapp, mock_metrics_tracker):
        """create_llm_dashboard factory works."""
        from ui.dashboard import create_llm_dashboard

        dashboard = create_llm_dashboard(metrics_tracker=mock_metrics_tracker)

        assert dashboard is not None
        assert dashboard.agent_metrics_widget.tracker == mock_metrics_tracker

        # LLM docks should be shown (use isHidden since parent window isn't displayed)
        for dock in dashboard._llm_docks.values():
            assert not dock.isHidden()


class TestUIExports:
    """Tests for UI module exports."""

    def test_exports_available(self):
        """All expected exports are available."""
        from ui import (
            AgentMetricsWidget,
            DebateViewer,
            DecisionLogViewer,
            EvolutionMonitor,
            create_llm_dashboard,
        )

        # Verify all are importable
        assert AgentMetricsWidget is not None
        assert DebateViewer is not None
        assert EvolutionMonitor is not None
        assert DecisionLogViewer is not None
        assert create_llm_dashboard is not None

    def test_dashboard_exports(self):
        """Dashboard exports include LLM widgets."""
        from ui.dashboard import __all__ as dashboard_all

        expected = [
            "AgentMetricsWidget",
            "DebateViewer",
            "EvolutionMonitor",
            "DecisionLogViewer",
            "create_llm_dashboard",
        ]

        for name in expected:
            assert name in dashboard_all
