"""
Tests for Walk-Forward Analysis Module

UPGRADE-015 Phase 9: Backtesting Robustness

Tests cover:
- Window generation methods
- Analysis execution
- Result aggregation
- Factory function
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtesting.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardMethod,
    WalkForwardResult,
    WindowResult,
    create_walk_forward_analyzer,
)


class TestWalkForwardConfig:
    """Test WalkForwardConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WalkForwardConfig()

        assert config.method == WalkForwardMethod.ROLLING
        assert config.in_sample_days == 252
        assert config.out_sample_days == 63
        assert config.num_windows == 5
        assert config.min_performance_ratio == 0.5
        assert config.robustness_threshold == 0.7

    def test_custom_config(self):
        """Test custom configuration."""
        config = WalkForwardConfig(
            method=WalkForwardMethod.ANCHORED,
            in_sample_days=180,
            out_sample_days=30,
            num_windows=10,
        )

        assert config.method == WalkForwardMethod.ANCHORED
        assert config.in_sample_days == 180
        assert config.out_sample_days == 30
        assert config.num_windows == 10


class TestWindowResult:
    """Test WindowResult dataclass."""

    def test_window_result_creation(self):
        """Test creating a window result."""
        result = WindowResult(
            window_id=0,
            in_sample_start=datetime(2020, 1, 1),
            in_sample_end=datetime(2021, 1, 1),
            out_sample_start=datetime(2021, 1, 1),
            out_sample_end=datetime(2021, 4, 1),
            in_sample_return=0.15,
            out_sample_return=0.08,
            performance_ratio=0.53,
        )

        assert result.window_id == 0
        assert result.in_sample_return == 0.15
        assert result.out_sample_return == 0.08
        assert result.performance_ratio == 0.53


class TestWalkForwardResult:
    """Test WalkForwardResult dataclass."""

    def test_to_dict(self):
        """Test converting result to dictionary."""
        window = WindowResult(
            window_id=0,
            in_sample_start=datetime(2020, 1, 1),
            in_sample_end=datetime(2021, 1, 1),
            out_sample_start=datetime(2021, 1, 1),
            out_sample_end=datetime(2021, 4, 1),
            in_sample_return=0.15,
            out_sample_return=0.08,
            performance_ratio=0.53,
        )

        result = WalkForwardResult(
            method=WalkForwardMethod.ROLLING,
            num_windows=1,
            window_results=[window],
            avg_in_sample_return=0.15,
            avg_out_sample_return=0.08,
            avg_performance_ratio=0.53,
            is_robust=False,
            robustness_score=0.65,
        )

        data = result.to_dict()

        assert data["method"] == "rolling"
        assert data["num_windows"] == 1
        assert data["avg_in_sample_return"] == 0.15
        assert data["robustness_score"] == 0.65
        assert len(data["windows"]) == 1


class TestWalkForwardAnalyzer:
    """Test WalkForwardAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = WalkForwardAnalyzer()

        assert analyzer.config.method == WalkForwardMethod.ROLLING
        assert analyzer.config.num_windows == 5

    def test_custom_config_initialization(self):
        """Test analyzer with custom config."""
        config = WalkForwardConfig(
            method=WalkForwardMethod.ANCHORED,
            num_windows=3,
        )
        analyzer = WalkForwardAnalyzer(config)

        assert analyzer.config.method == WalkForwardMethod.ANCHORED
        assert analyzer.config.num_windows == 3

    def test_generate_rolling_windows(self):
        """Test rolling window generation."""
        config = WalkForwardConfig(
            method=WalkForwardMethod.ROLLING,
            in_sample_days=100,
            out_sample_days=25,
            num_windows=3,
        )
        analyzer = WalkForwardAnalyzer(config)

        start = datetime(2020, 1, 1)
        end = datetime(2022, 1, 1)

        windows = analyzer.generate_windows(start, end)

        assert len(windows) == 3

        # First window
        assert windows[0][0] == start  # in_sample_start
        assert windows[0][1] == start + timedelta(days=100)  # in_sample_end

        # Windows should step forward by out_sample_days
        assert windows[1][0] == start + timedelta(days=25)

    def test_generate_anchored_windows(self):
        """Test anchored window generation."""
        config = WalkForwardConfig(
            method=WalkForwardMethod.ANCHORED,
            in_sample_days=100,
            out_sample_days=25,
            num_windows=3,
        )
        analyzer = WalkForwardAnalyzer(config)

        start = datetime(2020, 1, 1)
        end = datetime(2022, 1, 1)

        windows = analyzer.generate_windows(start, end)

        # All windows should start from the same date
        for window in windows:
            assert window[0] == start

    def test_generate_expanding_windows(self):
        """Test expanding window generation."""
        config = WalkForwardConfig(
            method=WalkForwardMethod.EXPANDING,
            in_sample_days=100,
            out_sample_days=25,
            num_windows=3,
        )
        analyzer = WalkForwardAnalyzer(config)

        start = datetime(2020, 1, 1)
        end = datetime(2022, 1, 1)

        windows = analyzer.generate_windows(start, end)

        # In-sample periods should grow
        assert windows[1][1] > windows[0][1]
        assert windows[2][1] > windows[1][1]

    def test_analyze_empty_returns(self):
        """Test analysis with empty returns."""
        analyzer = WalkForwardAnalyzer()
        result = analyzer.analyze([])

        assert result.num_windows == 0
        assert len(result.warnings) > 0

    def test_analyze_with_returns(self):
        """Test analysis with synthetic returns."""
        config = WalkForwardConfig(
            in_sample_days=50,
            out_sample_days=10,
            num_windows=3,
        )
        analyzer = WalkForwardAnalyzer(config)

        # Generate synthetic returns (200 days)
        base_date = datetime(2020, 1, 1)
        returns = [(base_date + timedelta(days=i), 0.001 * (1 + (i % 10) / 100)) for i in range(200)]

        result = analyzer.analyze(returns)

        assert result.num_windows > 0
        assert result.avg_in_sample_return != 0
        assert result.robustness_score >= 0

    def test_analyze_with_optimizer(self):
        """Test analysis with custom optimizer."""
        config = WalkForwardConfig(
            in_sample_days=30,
            out_sample_days=10,
            num_windows=2,
        )
        analyzer = WalkForwardAnalyzer(config)

        # Simple optimizer that returns fixed params
        def optimizer(returns):
            return {"threshold": 0.5}

        # Simple evaluator
        def evaluator(returns, params):
            ret_values = [r[1] for r in returns]
            total = sum(ret_values)
            return {"return": total, "sharpe": total * 10}

        base_date = datetime(2020, 1, 1)
        returns = [(base_date + timedelta(days=i), 0.001) for i in range(100)]

        result = analyzer.analyze(returns, optimizer=optimizer, evaluator=evaluator)

        # Check params were captured
        for window in result.window_results:
            assert "threshold" in window.optimal_params

    def test_default_evaluate(self):
        """Test default evaluation function."""
        analyzer = WalkForwardAnalyzer()

        returns = [(datetime(2020, 1, i), 0.01) for i in range(1, 21)]

        metrics = analyzer._default_evaluate(returns)

        assert "return" in metrics
        assert "sharpe" in metrics
        assert metrics["return"] > 0

    def test_aggregate_results_empty(self):
        """Test aggregation with no results."""
        analyzer = WalkForwardAnalyzer()
        result = analyzer._aggregate_results([], [])

        assert result.num_windows == 0
        assert len(result.warnings) > 0

    def test_calculate_param_stability(self):
        """Test parameter stability calculation."""
        analyzer = WalkForwardAnalyzer()

        # Identical params = perfect stability
        params = [{"a": 1, "b": 2}] * 5
        stability = analyzer._calculate_param_stability(params)
        assert stability == 1.0

        # Varying params
        params = [{"a": 1}, {"a": 2}, {"a": 3}]
        stability = analyzer._calculate_param_stability(params)
        assert 0 < stability < 1

    def test_robustness_scoring(self):
        """Test robustness score calculation."""
        config = WalkForwardConfig(
            in_sample_days=30,
            out_sample_days=10,
            num_windows=3,
            min_performance_ratio=0.5,
        )
        analyzer = WalkForwardAnalyzer(config)

        # Create windows with good performance ratios
        windows = [
            WindowResult(
                window_id=i,
                in_sample_start=datetime(2020, 1, 1),
                in_sample_end=datetime(2020, 2, 1),
                out_sample_start=datetime(2020, 2, 1),
                out_sample_end=datetime(2020, 2, 10),
                in_sample_return=0.10,
                out_sample_return=0.08,
                performance_ratio=0.8,
            )
            for i in range(3)
        ]

        result = analyzer._aggregate_results(windows, [{"a": 1}] * 3)

        # Good performance ratios should give good robustness
        assert result.robustness_score > 0.5


class TestCreateWalkForwardAnalyzer:
    """Test factory function."""

    def test_create_with_defaults(self):
        """Test creating analyzer with defaults."""
        analyzer = create_walk_forward_analyzer()

        assert analyzer.config.method == WalkForwardMethod.ROLLING
        assert analyzer.config.in_sample_days == 252

    def test_create_with_custom_method(self):
        """Test creating analyzer with custom method."""
        analyzer = create_walk_forward_analyzer(method="anchored")

        assert analyzer.config.method == WalkForwardMethod.ANCHORED

    def test_create_with_custom_windows(self):
        """Test creating analyzer with custom window settings."""
        analyzer = create_walk_forward_analyzer(
            in_sample_days=180,
            out_sample_days=30,
            num_windows=10,
        )

        assert analyzer.config.in_sample_days == 180
        assert analyzer.config.out_sample_days == 30
        assert analyzer.config.num_windows == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
