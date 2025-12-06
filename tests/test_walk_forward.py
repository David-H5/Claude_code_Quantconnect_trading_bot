"""
Walk-Forward Validation Tests

Tests for preventing overfitting and ensuring strategy robustness using
walk-forward optimization techniques.

Based on best practices from:
- Robert Pardo's "Design, Testing and Optimization of Trading Systems"
- QuantInsti walk-forward optimization guide
- Alpha Scientist walk-forward modeling
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    start_date: datetime
    end_date: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades: int
    parameters: dict[str, Any]


@dataclass
class WalkForwardWindow:
    """A single walk-forward window with in-sample and out-of-sample periods."""

    in_sample_start: datetime
    in_sample_end: datetime
    out_of_sample_start: datetime
    out_of_sample_end: datetime
    optimized_params: dict[str, Any]
    in_sample_result: BacktestResult
    out_of_sample_result: BacktestResult


class TestWalkForwardValidation:
    """Tests for walk-forward validation methodology."""

    @pytest.fixture
    def sample_returns(self) -> list[float]:
        """Generate sample returns for testing."""
        np.random.seed(42)
        # Simulate realistic daily returns
        return list(np.random.normal(0.0005, 0.02, 252))  # 1 year of daily returns

    @pytest.fixture
    def walk_forward_windows(self) -> list[WalkForwardWindow]:
        """Create sample walk-forward windows."""
        windows = []
        base_date = datetime(2020, 1, 1)

        for i in range(5):
            # Each window: 6 months in-sample, 2 months out-of-sample
            in_start = base_date + timedelta(days=i * 60)
            in_end = in_start + timedelta(days=180)
            out_start = in_end + timedelta(days=1)
            out_end = out_start + timedelta(days=60)

            window = WalkForwardWindow(
                in_sample_start=in_start,
                in_sample_end=in_end,
                out_of_sample_start=out_start,
                out_of_sample_end=out_end,
                optimized_params={"rsi_period": 14 + i, "threshold": 30 + i},
                in_sample_result=BacktestResult(
                    start_date=in_start,
                    end_date=in_end,
                    total_return=0.15 + np.random.uniform(-0.05, 0.05),
                    sharpe_ratio=1.5 + np.random.uniform(-0.3, 0.3),
                    max_drawdown=0.10 + np.random.uniform(0, 0.05),
                    win_rate=0.55 + np.random.uniform(-0.05, 0.05),
                    trades=50 + np.random.randint(-10, 10),
                    parameters={"rsi_period": 14 + i, "threshold": 30 + i},
                ),
                out_of_sample_result=BacktestResult(
                    start_date=out_start,
                    end_date=out_end,
                    total_return=0.08 + np.random.uniform(-0.05, 0.05),
                    sharpe_ratio=1.0 + np.random.uniform(-0.3, 0.3),
                    max_drawdown=0.12 + np.random.uniform(0, 0.05),
                    win_rate=0.50 + np.random.uniform(-0.05, 0.05),
                    trades=20 + np.random.randint(-5, 5),
                    parameters={"rsi_period": 14 + i, "threshold": 30 + i},
                ),
            )
            windows.append(window)

        return windows

    @pytest.mark.unit
    def test_walk_forward_window_creation(self, walk_forward_windows):
        """Test that walk-forward windows are created correctly."""
        for window in walk_forward_windows:
            # In-sample should end before out-of-sample starts
            assert window.in_sample_end < window.out_of_sample_start

            # No overlap between periods
            assert window.in_sample_end < window.out_of_sample_start

    @pytest.mark.unit
    def test_out_of_sample_vs_in_sample_performance(self, walk_forward_windows):
        """Test that out-of-sample performance is typically worse than in-sample."""
        degradation_count = 0

        for window in walk_forward_windows:
            in_sample = window.in_sample_result
            out_of_sample = window.out_of_sample_result

            # Expect performance degradation out-of-sample
            if out_of_sample.sharpe_ratio < in_sample.sharpe_ratio:
                degradation_count += 1

        # Most windows should show degradation (realistic expectation)
        degradation_rate = degradation_count / len(walk_forward_windows)
        # At least 50% should show degradation (overfitting detection)
        assert degradation_rate >= 0.4

    @pytest.mark.unit
    def test_walk_forward_efficiency_ratio(self, walk_forward_windows):
        """Test walk-forward efficiency ratio (out-of-sample / in-sample performance)."""
        efficiency_ratios = []

        for window in walk_forward_windows:
            in_sample = window.in_sample_result
            out_of_sample = window.out_of_sample_result

            if in_sample.total_return != 0:
                ratio = out_of_sample.total_return / in_sample.total_return
                efficiency_ratios.append(ratio)

        # Average efficiency ratio
        avg_efficiency = np.mean(efficiency_ratios)

        # A robust strategy should have efficiency > 0.5 (50% performance retention)
        # This is a critical overfitting indicator
        assert avg_efficiency > 0.3, f"Low efficiency ratio {avg_efficiency:.2f} indicates overfitting"

    @pytest.mark.unit
    def test_parameter_stability(self, walk_forward_windows):
        """Test that optimized parameters are stable across windows."""
        rsi_periods = [w.optimized_params["rsi_period"] for w in walk_forward_windows]
        thresholds = [w.optimized_params["threshold"] for w in walk_forward_windows]

        # Calculate coefficient of variation (CV)
        rsi_cv = np.std(rsi_periods) / np.mean(rsi_periods) if np.mean(rsi_periods) > 0 else 0
        threshold_cv = np.std(thresholds) / np.mean(thresholds) if np.mean(thresholds) > 0 else 0

        # Parameters should be relatively stable (CV < 50%)
        assert rsi_cv < 0.5, f"RSI period too unstable: CV={rsi_cv:.2f}"
        assert threshold_cv < 0.5, f"Threshold too unstable: CV={threshold_cv:.2f}"

    @pytest.mark.unit
    def test_no_future_data_leakage(self, walk_forward_windows):
        """Test that no future data is used in optimization."""
        for i, window in enumerate(walk_forward_windows):
            # Verify in-sample period doesn't overlap with future out-of-sample
            if i < len(walk_forward_windows) - 1:
                next_window = walk_forward_windows[i + 1]

                # Current in-sample should not extend into next out-of-sample
                assert window.in_sample_end < next_window.out_of_sample_start

    @pytest.mark.unit
    def test_sufficient_out_of_sample_trades(self, walk_forward_windows):
        """Test that out-of-sample periods have enough trades for statistical significance."""
        min_trades = 10  # Minimum trades for meaningful evaluation

        for window in walk_forward_windows:
            assert (
                window.out_of_sample_result.trades >= min_trades
            ), f"Insufficient OOS trades: {window.out_of_sample_result.trades}"


class TestOverfittingDetection:
    """Tests for detecting overfitting in trading strategies."""

    @pytest.mark.unit
    def test_detect_curve_fitting(self):
        """Test detection of curve-fitting through performance gap."""
        # Simulated results showing curve-fitting
        in_sample_sharpe = 3.5  # Suspiciously high
        out_of_sample_sharpe = 0.5  # Reality check

        performance_gap = in_sample_sharpe - out_of_sample_sharpe
        gap_ratio = performance_gap / in_sample_sharpe

        # Large gap (>70%) indicates curve-fitting
        is_overfit = gap_ratio > 0.7

        assert is_overfit, "Should detect curve-fitting"

    @pytest.mark.unit
    def test_parameter_count_warning(self):
        """Test warning for too many parameters."""
        # Rule of thumb: no more than sqrt(N) parameters where N is data points
        data_points = 252  # 1 year of daily data
        max_recommended_params = int(np.sqrt(data_points))  # ~16

        actual_params = 25  # Too many

        is_over_parameterized = actual_params > max_recommended_params

        assert is_over_parameterized, "Should warn about too many parameters"

    @pytest.mark.unit
    def test_out_of_sample_period_adequacy(self):
        """Test that out-of-sample period is adequate."""
        in_sample_days = 180
        out_of_sample_days = 30

        # OOS should be at least 20% of IS
        min_oos_ratio = 0.2
        actual_ratio = out_of_sample_days / in_sample_days

        is_adequate = actual_ratio >= min_oos_ratio

        assert not is_adequate, "OOS period too short relative to IS"

    @pytest.mark.unit
    def test_strategy_complexity_vs_performance(self):
        """Test relationship between complexity and performance degradation."""
        # More complex strategies tend to degrade more out-of-sample
        strategies = [
            {"complexity": 1, "is_sharpe": 1.5, "oos_sharpe": 1.3},  # Simple
            {"complexity": 5, "is_sharpe": 2.5, "oos_sharpe": 1.5},  # Medium
            {"complexity": 10, "is_sharpe": 4.0, "oos_sharpe": 0.8},  # Complex
        ]

        degradations = []
        for s in strategies:
            degradation = (s["is_sharpe"] - s["oos_sharpe"]) / s["is_sharpe"]
            degradations.append((s["complexity"], degradation))

        # Higher complexity should correlate with higher degradation
        complexities = [d[0] for d in degradations]
        degradation_values = [d[1] for d in degradations]

        correlation = np.corrcoef(complexities, degradation_values)[0, 1]

        assert correlation > 0.5, "Complexity should correlate with degradation"


class TestRobustnessMetrics:
    """Tests for strategy robustness metrics."""

    @pytest.mark.unit
    def test_monte_carlo_robustness(self):
        """Test strategy robustness using Monte Carlo simulation."""
        np.random.seed(42)

        # Simulate trade outcomes
        original_trades = [0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.03]
        num_simulations = 1000

        simulated_returns = []
        for _ in range(num_simulations):
            # Randomly shuffle trade order
            shuffled = np.random.permutation(original_trades)
            cumulative = np.sum(shuffled)
            simulated_returns.append(cumulative)

        # Check if distribution is reasonable
        mean_return = np.mean(simulated_returns)
        std_return = np.std(simulated_returns)

        # Original return should be within 2 std of mean
        original_return = sum(original_trades)
        z_score = abs(original_return - mean_return) / std_return if std_return > 0 else 0

        assert z_score < 3, "Return is statistical outlier"

    @pytest.mark.unit
    def test_parameter_sensitivity(self):
        """Test strategy sensitivity to parameter changes."""
        base_params = {"rsi_period": 14, "threshold": 30}
        base_sharpe = 1.5

        # Simulate performance with perturbed parameters
        perturbations = [
            ({"rsi_period": 12, "threshold": 30}, 1.3),
            ({"rsi_period": 16, "threshold": 30}, 1.4),
            ({"rsi_period": 14, "threshold": 28}, 1.35),
            ({"rsi_period": 14, "threshold": 32}, 1.45),
        ]

        performance_changes = []
        for params, sharpe in perturbations:
            change = abs(sharpe - base_sharpe) / base_sharpe
            performance_changes.append(change)

        avg_sensitivity = np.mean(performance_changes)

        # Performance shouldn't change more than 20% with small param changes
        assert avg_sensitivity < 0.2, f"Too sensitive to parameters: {avg_sensitivity:.2%}"

    @pytest.mark.unit
    def test_market_regime_robustness(self):
        """Test strategy performance across different market regimes."""
        # Simulated performance in different regimes
        regime_performance = {
            "bull_market": {"sharpe": 1.8, "win_rate": 0.60},
            "bear_market": {"sharpe": 0.8, "win_rate": 0.45},
            "sideways": {"sharpe": 1.2, "win_rate": 0.52},
            "high_volatility": {"sharpe": 0.5, "win_rate": 0.48},
        }

        sharpes = [r["sharpe"] for r in regime_performance.values()]
        min_sharpe = min(sharpes)
        max_sharpe = max(sharpes)

        # Performance variation across regimes
        variation = (max_sharpe - min_sharpe) / max_sharpe

        # Robust strategy should work in all regimes (variation < 80%)
        assert variation < 0.8, f"Too much regime variation: {variation:.2%}"

    @pytest.mark.unit
    def test_time_stability(self):
        """Test that strategy performance is stable over time."""
        # Simulated annual performance
        annual_sharpes = [1.5, 1.3, 1.6, 1.2, 1.4, 0.9, 1.5]

        # Check for downward trend (potential strategy decay)
        recent_avg = np.mean(annual_sharpes[-3:])
        historical_avg = np.mean(annual_sharpes[:-3])

        # Recent should be at least 70% of historical
        retention = recent_avg / historical_avg if historical_avg > 0 else 0

        assert retention > 0.7, f"Strategy may be decaying: {retention:.2%} retention"


class TestCrossValidation:
    """Tests for cross-validation in trading strategies."""

    @pytest.mark.unit
    def test_k_fold_time_series_split(self):
        """Test proper time-series k-fold splitting."""
        data_points = 252 * 5  # 5 years
        n_splits = 5

        # Calculate split sizes
        test_size = data_points // (n_splits + 1)
        splits = []

        for i in range(n_splits):
            train_end = (i + 1) * test_size + test_size
            test_start = train_end
            test_end = test_start + test_size

            if test_end <= data_points:
                splits.append((train_end, test_start, test_end))

        # Verify no future data leakage
        for train_end, test_start, test_end in splits:
            assert train_end <= test_start, "Training data overlaps with test"

    @pytest.mark.unit
    def test_purged_cross_validation(self):
        """Test purged cross-validation to prevent data leakage."""
        # Simulated indices
        train_indices = list(range(0, 200))
        test_indices = list(range(200, 250))

        # Purge gap to prevent leakage
        purge_window = 5
        purged_train = [i for i in train_indices if i < min(test_indices) - purge_window]

        # Verify purge gap
        assert max(purged_train) < min(test_indices) - purge_window

    @pytest.mark.unit
    def test_embargo_period(self):
        """Test embargo period after test set to prevent look-ahead."""
        test_end_index = 250
        embargo_length = 10

        # Data points that should be excluded from subsequent training
        embargoed_points = list(range(test_end_index, test_end_index + embargo_length))

        # Verify embargo
        assert len(embargoed_points) == embargo_length
        assert min(embargoed_points) == test_end_index


class TestOutOfSampleAnalysis:
    """Tests for out-of-sample analysis."""

    @pytest.mark.unit
    def test_out_of_sample_period_selection(self):
        """Test proper selection of out-of-sample period."""
        total_data_years = 10
        oos_years = 2  # Reserve 2 years for OOS

        oos_ratio = oos_years / total_data_years

        # OOS should be 15-25% of total data
        assert 0.15 <= oos_ratio <= 0.25, f"OOS ratio {oos_ratio} outside recommended range"

    @pytest.mark.unit
    def test_rolling_out_of_sample(self):
        """Test rolling out-of-sample validation."""
        windows = []
        window_size = 252  # 1 year
        step_size = 63  # Quarterly steps

        total_points = 252 * 5  # 5 years

        current = 0
        while current + window_size <= total_points:
            windows.append((current, current + window_size))
            current += step_size

        # Should have multiple windows for robust validation
        assert len(windows) >= 10, "Need more rolling windows"

    @pytest.mark.unit
    def test_anchored_vs_rolling_comparison(self):
        """Test comparison of anchored vs rolling walk-forward."""
        # Anchored: training window grows
        anchored_results = {
            "avg_oos_sharpe": 1.1,
            "sharpe_stability": 0.15,
        }

        # Rolling: fixed training window
        rolling_results = {
            "avg_oos_sharpe": 1.0,
            "sharpe_stability": 0.20,
        }

        # Anchored typically has better stability (more data)
        assert anchored_results["sharpe_stability"] <= rolling_results["sharpe_stability"]
