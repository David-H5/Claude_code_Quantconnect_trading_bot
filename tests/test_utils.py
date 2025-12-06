"""
Tests for utility calculation functions.
"""

import pytest

from utils.calculations import (
    calculate_cagr,
    calculate_kelly_fraction,
    calculate_max_drawdown,
    calculate_position_size,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_volatility,
    calculate_win_rate,
)


class TestPositionSizing:
    """Tests for position sizing calculations."""

    @pytest.mark.unit
    def test_basic_position_size(self):
        """Test basic position size calculation."""
        # $100k portfolio, 2% risk, entry $50, stop $48
        shares = calculate_position_size(
            portfolio_value=100000,
            risk_per_trade=0.02,
            entry_price=50.0,
            stop_loss_price=48.0,
        )
        # Risk = $2000, Risk per share = $2, Shares = 1000
        assert shares == 1000

    @pytest.mark.unit
    def test_position_size_with_max_constraint(self):
        """Test position size respects max position constraint."""
        shares = calculate_position_size(
            portfolio_value=100000,
            risk_per_trade=0.10,  # 10% risk would give large position
            entry_price=50.0,
            stop_loss_price=49.0,
            max_position_pct=0.5,  # But limit to 50% of portfolio
        )
        # Max shares = $50k / $50 = 1000
        assert shares <= 1000

    @pytest.mark.unit
    def test_position_size_short(self):
        """Test position sizing for short position."""
        shares = calculate_position_size(
            portfolio_value=100000,
            risk_per_trade=0.02,
            entry_price=50.0,
            stop_loss_price=52.0,  # Stop above entry for short
        )
        assert shares == 1000

    @pytest.mark.unit
    def test_position_size_invalid_inputs(self):
        """Test position sizing with invalid inputs."""
        with pytest.raises(ValueError):
            calculate_position_size(0, 0.02, 50, 48)

        with pytest.raises(ValueError):
            calculate_position_size(100000, 1.5, 50, 48)  # Risk > 100%

        with pytest.raises(ValueError):
            calculate_position_size(100000, 0.02, 50, 50)  # Same entry and stop


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    @pytest.mark.unit
    def test_sharpe_positive_returns(self):
        """Test Sharpe ratio with positive returns."""
        returns = [0.01, 0.02, 0.01, 0.015, 0.01]  # Consistent positive
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe is not None
        assert sharpe > 0

    @pytest.mark.unit
    def test_sharpe_mixed_returns(self):
        """Test Sharpe ratio with mixed returns."""
        returns = [0.02, -0.01, 0.03, -0.02, 0.01]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe is not None

    @pytest.mark.unit
    def test_sharpe_insufficient_data(self):
        """Test Sharpe ratio with insufficient data."""
        assert calculate_sharpe_ratio([0.01]) is None
        assert calculate_sharpe_ratio([]) is None

    @pytest.mark.unit
    def test_sharpe_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = [0.01, 0.01, 0.01]  # No variance
        assert calculate_sharpe_ratio(returns) is None


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    @pytest.mark.unit
    def test_drawdown_calculation(self):
        """Test basic drawdown calculation."""
        equity = [100, 110, 105, 95, 100, 90, 95]
        dd, peak_idx, trough_idx = calculate_max_drawdown(equity)

        # Max DD is from 110 to 90 = 18.18%
        assert dd == pytest.approx(0.1818, rel=0.01)
        assert peak_idx == 1  # Peak at 110
        assert trough_idx == 5  # Trough at 90

    @pytest.mark.unit
    def test_no_drawdown(self):
        """Test with no drawdown (always increasing)."""
        equity = [100, 110, 120, 130]
        dd, peak_idx, trough_idx = calculate_max_drawdown(equity)
        assert dd == 0.0

    @pytest.mark.unit
    def test_drawdown_insufficient_data(self):
        """Test with insufficient data."""
        dd, _, _ = calculate_max_drawdown([100])
        assert dd == 0.0


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    @pytest.mark.unit
    def test_sortino_basic(self):
        """Test basic Sortino calculation."""
        returns = [0.02, -0.01, 0.03, -0.02, 0.01]
        sortino = calculate_sortino_ratio(returns)
        assert sortino is not None

    @pytest.mark.unit
    def test_sortino_no_downside(self):
        """Test Sortino with no downside returns."""
        returns = [0.01, 0.02, 0.03]  # All positive
        assert calculate_sortino_ratio(returns) is None


class TestWinRate:
    """Tests for win rate calculation."""

    @pytest.mark.unit
    def test_win_rate_calculation(self):
        """Test basic win rate calculation."""
        trades = [100, -50, 75, -25, 50]
        assert calculate_win_rate(trades) == 0.6  # 3 wins out of 5

    @pytest.mark.unit
    def test_win_rate_all_wins(self):
        """Test win rate with all winning trades."""
        trades = [100, 50, 75]
        assert calculate_win_rate(trades) == 1.0

    @pytest.mark.unit
    def test_win_rate_empty(self):
        """Test win rate with no trades."""
        assert calculate_win_rate([]) is None


class TestProfitFactor:
    """Tests for profit factor calculation."""

    @pytest.mark.unit
    def test_profit_factor_calculation(self):
        """Test basic profit factor calculation."""
        trades = [100, -50, 75, -25]
        pf = calculate_profit_factor(trades)
        # Gross profit = 175, Gross loss = 75
        assert pf == pytest.approx(175 / 75)

    @pytest.mark.unit
    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        trades = [100, 50, 75]
        assert calculate_profit_factor(trades) == float("inf")

    @pytest.mark.unit
    def test_profit_factor_empty(self):
        """Test profit factor with no trades."""
        assert calculate_profit_factor([]) is None


class TestCAGR:
    """Tests for CAGR calculation."""

    @pytest.mark.unit
    def test_cagr_calculation(self):
        """Test basic CAGR calculation."""
        cagr = calculate_cagr(100, 200, 5)  # Doubled in 5 years
        # (2)^(1/5) - 1 = 0.1487
        assert cagr == pytest.approx(0.1487, rel=0.01)

    @pytest.mark.unit
    def test_cagr_one_year(self):
        """Test CAGR for one year."""
        cagr = calculate_cagr(100, 110, 1)
        assert cagr == pytest.approx(0.10)

    @pytest.mark.unit
    def test_cagr_invalid(self):
        """Test CAGR with invalid inputs."""
        assert calculate_cagr(0, 100, 1) is None
        assert calculate_cagr(100, 0, 1) is None
        assert calculate_cagr(100, 200, 0) is None


class TestVolatility:
    """Tests for volatility calculation."""

    @pytest.mark.unit
    def test_volatility_calculation(self):
        """Test basic volatility calculation."""
        returns = [0.01, -0.01, 0.02, -0.02, 0.01]
        vol = calculate_volatility(returns)
        assert vol is not None
        assert vol > 0

    @pytest.mark.unit
    def test_volatility_insufficient_data(self):
        """Test volatility with insufficient data."""
        assert calculate_volatility([0.01]) is None


class TestKellyFraction:
    """Tests for Kelly criterion calculation."""

    @pytest.mark.unit
    def test_kelly_calculation(self):
        """Test basic Kelly calculation."""
        # 60% win rate, 1:1 win/loss ratio
        kelly = calculate_kelly_fraction(0.6, 1.0)
        # f = 0.6 - (0.4/1.0) = 0.2
        assert kelly == pytest.approx(0.2)

    @pytest.mark.unit
    def test_kelly_negative_edge(self):
        """Test Kelly with negative edge."""
        # 40% win rate, 1:1 ratio = negative edge
        kelly = calculate_kelly_fraction(0.4, 1.0)
        assert kelly == 0.0  # Don't bet

    @pytest.mark.unit
    def test_kelly_high_win_rate(self):
        """Test Kelly with high win rate."""
        kelly = calculate_kelly_fraction(0.7, 2.0)
        # f = 0.7 - (0.3/2.0) = 0.55
        assert kelly == pytest.approx(0.55)

    @pytest.mark.unit
    def test_kelly_invalid_inputs(self):
        """Test Kelly with invalid inputs."""
        assert calculate_kelly_fraction(0, 1.0) == 0.0
        assert calculate_kelly_fraction(0.5, 0) == 0.0
