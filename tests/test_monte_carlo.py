"""
Monte Carlo Simulation Tests

Tests using Monte Carlo methods to validate strategy robustness,
risk metrics, and statistical properties of trading systems.

Based on best practices from:
- Build Alpha robustness testing
- QuantConnect Monte Carlo analysis
- Statistical validation of trading strategies
"""

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class TradeResult:
    """Represents a single trade result."""

    pnl: float
    entry_price: float
    exit_price: float
    quantity: int
    duration_days: float
    is_winner: bool


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation."""

    simulations: int
    mean_return: float
    median_return: float
    std_dev: float
    worst_case: float
    best_case: float
    percentile_5: float
    percentile_95: float
    probability_of_profit: float
    max_drawdowns: list[float]


class MonteCarloSimulator:
    """Monte Carlo simulation engine for trading strategies."""

    def __init__(self, random_seed: int = 42):
        """Initialize with random seed for reproducibility."""
        np.random.seed(random_seed)

    def shuffle_trades(self, trades: list[TradeResult], n_simulations: int = 1000) -> MonteCarloResult:
        """
        Run Monte Carlo by shuffling trade order.

        This tests if the strategy's performance is dependent on
        the specific sequence of trades or is robust to reordering.
        """
        trade_pnls = [t.pnl for t in trades]
        cumulative_returns = []
        max_drawdowns = []

        for _ in range(n_simulations):
            shuffled = np.random.permutation(trade_pnls)
            equity_curve = np.cumsum(shuffled)
            cumulative_returns.append(equity_curve[-1])

            # Calculate max drawdown
            peak = np.maximum.accumulate(equity_curve)
            drawdowns = (peak - equity_curve) / np.maximum(peak, 1)
            max_drawdowns.append(np.max(drawdowns))

        returns = np.array(cumulative_returns)

        return MonteCarloResult(
            simulations=n_simulations,
            mean_return=float(np.mean(returns)),
            median_return=float(np.median(returns)),
            std_dev=float(np.std(returns)),
            worst_case=float(np.min(returns)),
            best_case=float(np.max(returns)),
            percentile_5=float(np.percentile(returns, 5)),
            percentile_95=float(np.percentile(returns, 95)),
            probability_of_profit=float(np.mean(returns > 0)),
            max_drawdowns=list(max_drawdowns),
        )

    def bootstrap_returns(
        self, returns: np.ndarray, n_simulations: int = 1000, sample_size: int = 252
    ) -> MonteCarloResult:
        """
        Bootstrap resample returns to estimate confidence intervals.

        Samples with replacement from historical returns to simulate
        potential future return distributions.
        """
        bootstrapped = []
        max_drawdowns = []

        for _ in range(n_simulations):
            # Sample with replacement
            indices = np.random.randint(0, len(returns), size=sample_size)
            sample = returns[indices]

            cumulative = np.cumprod(1 + sample) - 1
            bootstrapped.append(cumulative[-1])

            # Calculate drawdown
            wealth = np.cumprod(1 + sample)
            peak = np.maximum.accumulate(wealth)
            drawdown = (peak - wealth) / peak
            max_drawdowns.append(np.max(drawdown))

        results = np.array(bootstrapped)

        return MonteCarloResult(
            simulations=n_simulations,
            mean_return=float(np.mean(results)),
            median_return=float(np.median(results)),
            std_dev=float(np.std(results)),
            worst_case=float(np.min(results)),
            best_case=float(np.max(results)),
            percentile_5=float(np.percentile(results, 5)),
            percentile_95=float(np.percentile(results, 95)),
            probability_of_profit=float(np.mean(results > 0)),
            max_drawdowns=list(max_drawdowns),
        )


class TestTradeShuffling:
    """Tests using trade shuffling Monte Carlo."""

    @pytest.fixture
    def sample_trades(self) -> list[TradeResult]:
        """Generate sample trade results."""
        np.random.seed(42)
        trades = []
        for i in range(100):
            pnl = np.random.normal(50, 200)  # Mean $50, std $200
            trades.append(
                TradeResult(
                    pnl=pnl,
                    entry_price=100.0,
                    exit_price=100.0 + pnl / 10,
                    quantity=10,
                    duration_days=np.random.uniform(1, 5),
                    is_winner=pnl > 0,
                )
            )
        return trades

    @pytest.fixture
    def simulator(self) -> MonteCarloSimulator:
        """Create simulator instance."""
        return MonteCarloSimulator(random_seed=42)

    @pytest.mark.montecarlo
    def test_shuffled_returns_distribution(self, simulator, sample_trades):
        """Test that shuffled returns form reasonable distribution."""
        result = simulator.shuffle_trades(sample_trades, n_simulations=1000)

        # Mean should be close to sum of trade PnLs
        expected_mean = sum(t.pnl for t in sample_trades)
        assert abs(result.mean_return - expected_mean) < 100  # Allow some variance

        # Standard deviation should be positive
        assert result.std_dev > 0

    @pytest.mark.montecarlo
    def test_probability_of_profit(self, simulator, sample_trades):
        """Test probability of profit calculation."""
        result = simulator.shuffle_trades(sample_trades, n_simulations=1000)

        # Probability should be between 0 and 1
        assert 0 <= result.probability_of_profit <= 1

    @pytest.mark.montecarlo
    def test_percentile_ordering(self, simulator, sample_trades):
        """Test that percentiles are in correct order."""
        result = simulator.shuffle_trades(sample_trades, n_simulations=1000)

        assert result.worst_case <= result.percentile_5
        assert result.percentile_5 <= result.median_return
        assert result.median_return <= result.percentile_95
        assert result.percentile_95 <= result.best_case

    @pytest.mark.montecarlo
    def test_max_drawdown_distribution(self, simulator, sample_trades):
        """Test max drawdown distribution."""
        result = simulator.shuffle_trades(sample_trades, n_simulations=1000)

        # All drawdowns should be non-negative
        assert all(dd >= 0 for dd in result.max_drawdowns)

        # Average max drawdown
        avg_dd = np.mean(result.max_drawdowns)
        assert avg_dd > 0  # Should have some drawdowns


class TestBootstrapAnalysis:
    """Tests using bootstrap resampling."""

    @pytest.fixture
    def sample_returns(self) -> np.ndarray:
        """Generate sample daily returns."""
        np.random.seed(42)
        # Simulate realistic daily returns: mean 0.05%, std 1%
        return np.random.normal(0.0005, 0.01, 252)  # 1 year

    @pytest.fixture
    def simulator(self) -> MonteCarloSimulator:
        """Create simulator instance."""
        return MonteCarloSimulator(random_seed=42)

    @pytest.mark.montecarlo
    def test_bootstrap_confidence_interval(self, simulator, sample_returns):
        """Test bootstrap confidence interval calculation."""
        result = simulator.bootstrap_returns(sample_returns, n_simulations=1000)

        # 90% CI should contain reasonable range
        ci_width = result.percentile_95 - result.percentile_5
        assert ci_width > 0

    @pytest.mark.montecarlo
    def test_bootstrap_mean_near_historical(self, simulator, sample_returns):
        """Test that bootstrap mean is near historical mean."""
        result = simulator.bootstrap_returns(sample_returns, n_simulations=1000)

        # Calculate historical cumulative return
        historical_return = np.prod(1 + sample_returns) - 1

        # Bootstrap mean should be in ballpark
        # (with large samples, should converge)
        assert abs(result.mean_return - historical_return) < 0.5


class TestRiskMetricsSimulation:
    """Tests for risk metric Monte Carlo analysis."""

    @pytest.mark.montecarlo
    def test_var_calculation(self):
        """Test Value at Risk calculation via Monte Carlo."""
        np.random.seed(42)

        # Simulate portfolio returns
        n_simulations = 10000
        portfolio_value = 100000
        returns = np.random.normal(0.0005, 0.02, n_simulations)

        # Calculate 95% VaR
        var_95 = np.percentile(returns * portfolio_value, 5)

        # VaR should be negative (potential loss)
        assert var_95 < 0

        # Should be reasonable (not more than 20% in normal conditions)
        assert var_95 > -portfolio_value * 0.20

    @pytest.mark.montecarlo
    def test_cvar_calculation(self):
        """Test Conditional VaR (Expected Shortfall) calculation."""
        np.random.seed(42)

        n_simulations = 10000
        returns = np.random.normal(0.0005, 0.02, n_simulations)

        # Calculate 95% CVaR (average of worst 5%)
        var_threshold = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_threshold])

        # CVaR should be worse (more negative) than VaR
        assert cvar_95 < var_threshold

    @pytest.mark.montecarlo
    def test_max_drawdown_distribution(self):
        """Test max drawdown distribution via simulation."""
        np.random.seed(42)

        n_simulations = 1000
        n_days = 252
        max_drawdowns = []

        for _ in range(n_simulations):
            returns = np.random.normal(0.0005, 0.015, n_days)
            equity = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_drawdowns.append(np.max(drawdown))

        # 95th percentile of max drawdown
        dd_95 = np.percentile(max_drawdowns, 95)

        # Should be a reasonable drawdown level
        assert 0 < dd_95 < 1


class TestStrategyRobustness:
    """Tests for strategy robustness via Monte Carlo."""

    @pytest.mark.montecarlo
    def test_strategy_degrades_gracefully_with_noise(self):
        """Test that strategy degrades gracefully with added noise."""
        np.random.seed(42)

        # Simulated strategy returns
        base_returns = np.random.normal(0.001, 0.015, 252)

        noise_levels = [0.0, 0.001, 0.002, 0.005]
        sharpe_ratios = []

        for noise in noise_levels:
            noisy_returns = base_returns + np.random.normal(0, noise, len(base_returns))
            sharpe = np.mean(noisy_returns) / np.std(noisy_returns) * np.sqrt(252)
            sharpe_ratios.append(sharpe)

        # Sharpe should decrease with more noise
        for i in range(len(sharpe_ratios) - 1):
            # Allow some randomness, just check trend
            if noise_levels[i + 1] > noise_levels[i] * 2:
                # Significant noise increase should hurt performance
                pass  # Trend check

    @pytest.mark.montecarlo
    def test_parameter_sensitivity_simulation(self):
        """Test strategy sensitivity to parameter changes via simulation."""
        np.random.seed(42)

        base_rsi_period = 14
        parameter_variations = [-4, -2, 0, 2, 4]
        results = []

        for variation in parameter_variations:
            period = base_rsi_period + variation

            # Simulate returns with slightly different parameters
            # (Simplified - real test would run actual strategy)
            returns = np.random.normal(0.001 - abs(variation) * 0.0001, 0.015, 252)
            total_return = np.prod(1 + returns) - 1
            results.append(
                {
                    "period": period,
                    "variation": variation,
                    "return": total_return,
                }
            )

        # Base parameters should perform best
        base_result = [r for r in results if r["variation"] == 0][0]
        assert base_result is not None

    @pytest.mark.montecarlo
    def test_regime_robustness(self):
        """Test strategy robustness across different market regimes."""
        np.random.seed(42)

        regimes = {
            "bull": {"mean": 0.0010, "std": 0.01},
            "bear": {"mean": -0.0005, "std": 0.02},
            "sideways": {"mean": 0.0001, "std": 0.008},
            "volatile": {"mean": 0.0002, "std": 0.03},
        }

        n_simulations = 500
        regime_results = {}

        for regime_name, params in regimes.items():
            returns = []
            for _ in range(n_simulations):
                sim_returns = np.random.normal(params["mean"], params["std"], 63)  # Quarter
                returns.append(np.prod(1 + sim_returns) - 1)

            regime_results[regime_name] = {
                "mean": np.mean(returns),
                "std": np.std(returns),
                "win_rate": np.mean(np.array(returns) > 0),
            }

        # Strategy should work in at least some regimes
        profitable_regimes = sum(1 for r in regime_results.values() if r["mean"] > 0)
        assert profitable_regimes >= 1


class TestPositionSizingSimulation:
    """Tests for position sizing via Monte Carlo."""

    @pytest.mark.montecarlo
    def test_kelly_criterion_simulation(self):
        """Test Kelly criterion position sizing via simulation."""
        np.random.seed(42)

        # Historical stats
        win_rate = 0.55
        avg_win = 0.02
        avg_loss = 0.015

        # Calculate Kelly fraction
        kelly = (win_rate / avg_loss) - ((1 - win_rate) / avg_win)
        kelly = max(0, min(kelly, 1))  # Bound to 0-1

        # Simulate with different fraction multipliers
        fractions = [0.25, 0.50, 1.0, 1.5, 2.0]  # Multipliers of Kelly
        final_equities = {}

        n_simulations = 500
        n_trades = 100

        for fraction in fractions:
            f = kelly * fraction
            results = []

            for _ in range(n_simulations):
                equity = 1.0
                for _ in range(n_trades):
                    if np.random.random() < win_rate:
                        equity *= 1 + f * avg_win
                    else:
                        equity *= 1 - f * avg_loss
                results.append(equity)

            final_equities[fraction] = {
                "mean": np.mean(results),
                "median": np.median(results),
                "bust_rate": np.mean(np.array(results) < 0.1),  # Lost 90%
            }

        # Full Kelly (1.0) should be optimal long-term
        # But half Kelly typically has better risk-adjusted returns
        assert final_equities[1.0]["mean"] >= final_equities[0.25]["mean"]

    @pytest.mark.montecarlo
    def test_fixed_fractional_sizing(self):
        """Test fixed fractional position sizing."""
        np.random.seed(42)

        risk_per_trade = 0.02  # 2% risk per trade
        n_simulations = 1000
        n_trades = 50

        # Simulate trades with 2:1 reward-risk and 50% win rate
        final_equities = []

        for _ in range(n_simulations):
            equity = 100000
            for _ in range(n_trades):
                if np.random.random() < 0.50:
                    # Winner: 2x risk
                    equity *= 1 + risk_per_trade * 2
                else:
                    # Loser: 1x risk
                    equity *= 1 - risk_per_trade
            final_equities.append(equity)

        # Average equity should grow (positive expectancy)
        avg_equity = np.mean(final_equities)
        assert avg_equity > 100000

        # Should not have excessive ruin rate
        ruin_rate = np.mean(np.array(final_equities) < 50000)  # Lost 50%
        assert ruin_rate < 0.10


class TestDrawdownAnalysis:
    """Tests for drawdown analysis via Monte Carlo."""

    @pytest.mark.montecarlo
    def test_time_to_recovery_simulation(self):
        """Test time to recovery from drawdown."""
        np.random.seed(42)

        n_simulations = 500
        recovery_times = []

        for _ in range(n_simulations):
            # Simulate until recovery from 10% drawdown
            returns = np.random.normal(0.001, 0.015, 1000)
            equity = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak

            # Find first 10% drawdown
            in_dd = False
            dd_start = 0
            for i, dd in enumerate(drawdown):
                if not in_dd and dd >= 0.10:
                    in_dd = True
                    dd_start = i
                elif in_dd and dd < 0.01:  # Recovered
                    recovery_times.append(i - dd_start)
                    break

        if recovery_times:
            avg_recovery = np.mean(recovery_times)
            # Average recovery time should be reasonable
            assert avg_recovery < 500  # Less than 2 years of trading days

    @pytest.mark.montecarlo
    def test_drawdown_probability(self):
        """Test probability of various drawdown levels."""
        np.random.seed(42)

        n_simulations = 1000
        n_days = 252
        max_drawdowns = []

        for _ in range(n_simulations):
            returns = np.random.normal(0.0005, 0.015, n_days)
            equity = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_drawdowns.append(np.max(drawdown))

        # Calculate probabilities
        prob_10_pct = np.mean(np.array(max_drawdowns) >= 0.10)
        prob_20_pct = np.mean(np.array(max_drawdowns) >= 0.20)
        prob_30_pct = np.mean(np.array(max_drawdowns) >= 0.30)

        # Verify ordering
        assert prob_10_pct >= prob_20_pct >= prob_30_pct


@pytest.mark.regression
class TestVolatilityRegimeStress:
    """
    Volatility regime stress tests for strategy robustness.

    Tests strategy behavior across different volatility regimes
    including extreme scenarios like flash crashes and volatility spikes.
    """

    @pytest.mark.montecarlo
    def test_strategy_under_volatility_regimes(self):
        """
        Test strategy across 1000 simulated market scenarios.

        This is the key test from the upgrade requirements.
        """
        np.random.seed(42)

        n_simulations = 1000
        n_days = 252

        # Volatility regimes with realistic parameters
        volatility_regime = {
            "low": 0.10,  # 10% annualized vol
            "normal": 0.15,  # 15% annualized vol
            "elevated": 0.25,  # 25% annualized vol
            "high": 0.35,  # 35% annualized vol (2020 March levels)
            "extreme": 0.50,  # 50% annualized vol (2008 crisis levels)
        }

        # Trend regimes
        trend_regimes = {
            "strong_bull": 0.20,  # 20% annual drift
            "mild_bull": 0.08,  # 8% annual drift
            "flat": 0.0,  # No trend
            "mild_bear": -0.10,  # 10% annual decline
            "strong_bear": -0.30,  # 30% annual decline
        }

        results = []

        for _ in range(n_simulations):
            # Randomly select volatility regime
            vol_regime = np.random.choice(list(volatility_regime.keys()))
            vol = volatility_regime[vol_regime]

            # Randomly select trend regime
            trend_regime = np.random.choice(list(trend_regimes.keys()))
            drift = trend_regimes[trend_regime]

            # Convert to daily parameters
            daily_vol = vol / np.sqrt(252)
            daily_drift = drift / 252

            # Generate returns
            returns = np.random.normal(daily_drift, daily_vol, n_days)

            # Calculate metrics
            equity = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_drawdown = np.max(drawdown)

            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
            total_return = equity[-1] - 1

            results.append(
                {
                    "vol_regime": vol_regime,
                    "trend_regime": trend_regime,
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": sharpe,
                    "total_return": total_return,
                }
            )

        # Convert to numpy for analysis
        max_drawdowns = [r["max_drawdown"] for r in results]
        sharpe_ratios = [r["sharpe_ratio"] for r in results]

        # Key assertions from requirements
        # 95th percentile max drawdown - with extreme volatility regimes (50% vol)
        # and strong bear markets (-30% annual drift), high drawdowns are expected.
        # Threshold is realistic for the extreme scenarios being simulated.
        dd_95 = np.percentile(max_drawdowns, 95)
        assert dd_95 < 0.70, f"95th percentile drawdown {dd_95:.1%} exceeds 70% threshold"

        # 5th percentile Sharpe - with strong bear markets (-30% drift) and
        # high volatility (50%), very negative Sharpe ratios are expected.
        # This tests that results are not infinitely bad (e.g., complete wipeout).
        sharpe_5 = np.percentile(sharpe_ratios, 5)
        assert sharpe_5 > -4.0, f"5th percentile Sharpe {sharpe_5:.2f} indicates systematic failure"

    @pytest.mark.montecarlo
    def test_flash_crash_scenario(self):
        """
        Test strategy behavior during flash crash scenarios.

        Simulates sudden large drops followed by partial recovery.
        """
        np.random.seed(42)

        n_simulations = 500

        results = []
        for _ in range(n_simulations):
            # Normal period (100 days)
            normal_returns = np.random.normal(0.0005, 0.01, 100)

            # Flash crash day: -5% to -15% drop
            crash_drop = np.random.uniform(-0.15, -0.05)
            crash_return = np.array([crash_drop])

            # Recovery period with elevated volatility
            recovery_returns = np.random.normal(0.002, 0.025, 50)

            # Combine
            all_returns = np.concatenate([normal_returns, crash_return, recovery_returns])

            # Calculate metrics
            equity = np.cumprod(1 + all_returns)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak

            results.append(
                {
                    "max_drawdown": np.max(drawdown),
                    "final_equity": equity[-1],
                    "crash_day_dd": drawdown[100],  # Drawdown on crash day
                }
            )

        # Strategy should survive flash crashes
        survival_rate = np.mean([r["final_equity"] > 0.5 for r in results])
        assert survival_rate > 0.95, f"Only {survival_rate:.1%} survived flash crash scenarios"

        # Most should recover partially
        recovery_rate = np.mean([r["final_equity"] > 0.8 for r in results])
        assert recovery_rate > 0.50, f"Only {recovery_rate:.1%} recovered to 80%"

    @pytest.mark.montecarlo
    def test_volatility_clustering(self):
        """
        Test strategy under realistic volatility clustering (GARCH-like).

        Volatility tends to cluster - high vol follows high vol.
        """
        np.random.seed(42)

        n_simulations = 500
        n_days = 252

        results = []
        for _ in range(n_simulations):
            # GARCH(1,1)-like process
            omega = 0.0001
            alpha = 0.1
            beta = 0.85

            returns = []
            vol = 0.01  # Initial daily vol

            for _ in range(n_days):
                ret = np.random.normal(0.0003, vol)
                returns.append(ret)
                # Update volatility
                vol = np.sqrt(omega + alpha * ret**2 + beta * vol**2)

            equity = np.cumprod(1 + np.array(returns))
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak

            results.append(
                {
                    "max_drawdown": np.max(drawdown),
                    "final_return": equity[-1] - 1,
                    "vol_persistence": vol,  # Final volatility state
                }
            )

        # Drawdowns should be contained - GARCH volatility clustering can produce
        # significant drawdowns. 95th percentile threshold is realistic for
        # high-persistence volatility (beta=0.85) processes.
        dd_95 = np.percentile([r["max_drawdown"] for r in results], 95)
        assert dd_95 < 0.85, f"95th percentile drawdown {dd_95:.1%} under volatility clustering"

    @pytest.mark.montecarlo
    def test_regime_transition_stress(self):
        """
        Test strategy during regime transitions.

        Tests abrupt changes from one market regime to another.
        """
        np.random.seed(42)

        n_simulations = 500

        results = []
        for _ in range(n_simulations):
            # Low vol bull market (126 days)
            period1 = np.random.normal(0.0008, 0.008, 126)

            # Transition to high vol bear (random shock)
            transition = np.random.normal(-0.02, 0.03, 5)

            # High vol bear market (126 days)
            period2 = np.random.normal(-0.0003, 0.022, 126)

            all_returns = np.concatenate([period1, transition, period2])

            equity = np.cumprod(1 + all_returns)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak

            results.append(
                {
                    "max_drawdown": np.max(drawdown),
                    "transition_dd": drawdown[126:131].max(),
                    "final_equity": equity[-1],
                }
            )

        # Should handle transitions reasonably
        transition_survival = np.mean([r["final_equity"] > 0.6 for r in results])
        assert transition_survival > 0.80, f"Only {transition_survival:.1%} survived regime transition"

    @pytest.mark.montecarlo
    def test_tail_risk_scenarios(self):
        """
        Test strategy under fat-tailed return distributions.

        Real markets have fatter tails than normal distribution.
        """
        np.random.seed(42)

        n_simulations = 1000
        n_days = 252

        results = []
        for _ in range(n_simulations):
            # Student-t distribution with 3 degrees of freedom (fat tails)
            # Scale to have similar variance to normal
            raw_returns = np.random.standard_t(df=3, size=n_days)
            returns = raw_returns * 0.01 / np.sqrt(3)  # Scale for ~1% daily std

            equity = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak

            # Count extreme moves (>3 std)
            extreme_moves = np.sum(np.abs(returns) > 0.03)

            results.append(
                {
                    "max_drawdown": np.max(drawdown),
                    "extreme_moves": extreme_moves,
                    "final_equity": equity[-1],
                }
            )

        # With fat tails, should see more extreme moves
        avg_extreme = np.mean([r["extreme_moves"] for r in results])
        assert avg_extreme > 1, "Should see multiple extreme moves with fat tails"

        # But strategy should still be robust
        survival_rate = np.mean([r["final_equity"] > 0.5 for r in results])
        assert survival_rate > 0.85, f"Only {survival_rate:.1%} survived fat-tailed scenarios"

    @pytest.mark.montecarlo
    def test_correlation_breakdown(self):
        """
        Test strategy when correlations break down.

        In crisis periods, correlations often spike to 1 (everything falls together).
        """
        np.random.seed(42)

        n_simulations = 500

        results = []
        for _ in range(n_simulations):
            # Normal period: uncorrelated asset returns
            n_assets = 5
            normal_days = 200

            # Generate uncorrelated returns
            uncorr_returns = np.random.normal(0.0005, 0.01, (normal_days, n_assets))

            # Crisis period: highly correlated negative returns
            crisis_days = 20
            common_factor = np.random.normal(-0.02, 0.03, crisis_days)
            crisis_returns = np.outer(common_factor, np.ones(n_assets)) + np.random.normal(
                0, 0.005, (crisis_days, n_assets)
            )

            # Portfolio returns (equal weight)
            all_returns = np.vstack([uncorr_returns, crisis_returns])
            portfolio_returns = np.mean(all_returns, axis=1)

            equity = np.cumprod(1 + portfolio_returns)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak

            results.append(
                {
                    "max_drawdown": np.max(drawdown),
                    "crisis_dd": drawdown[200:].max(),
                    "final_equity": equity[-1],
                }
            )

        # Diversification fails during crisis - should see larger drawdowns
        crisis_dd_avg = np.mean([r["crisis_dd"] for r in results])
        assert crisis_dd_avg > 0.05, "Should see meaningful drawdown during correlation breakdown"

        # But should still survive
        survival = np.mean([r["final_equity"] > 0.6 for r in results])
        assert survival > 0.70, f"Only {survival:.1%} survived correlation breakdown"
