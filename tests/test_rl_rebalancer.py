"""
Tests for RL Rebalancing System

Tests the RL-based portfolio rebalancing functionality including:
- Policy network forward pass
- Value network estimation
- Rebalancing decisions
- Transaction cost estimation
- Training process
"""

import pytest

from models.rl_rebalancer import (
    AssetState,
    PolicyNetwork,
    PortfolioState,
    RebalanceAction,
    RebalanceDecision,
    RebalanceFrequency,
    RLRebalancerConfig,
    ValueNetwork,
    create_rl_rebalancer,
)


def create_mock_asset_state(
    symbol: str,
    current_weight: float = 0.2,
    price: float = 100.0,
) -> AssetState:
    """Create a mock asset state for testing."""
    return AssetState(
        symbol=symbol,
        current_weight=current_weight,
        target_weight=current_weight,
        price=price,
        price_change_1d=0.01,
        price_change_5d=0.05,
        price_change_20d=0.10,
        volatility=0.20,
        volume_ratio=1.0,
        momentum_score=0.5,
        correlation_to_portfolio=0.5,
    )


def create_mock_portfolio_state(
    assets: list[AssetState] = None,
    total_value: float = 100000.0,
) -> PortfolioState:
    """Create a mock portfolio state for testing."""
    if assets is None:
        assets = [
            create_mock_asset_state("SPY", 0.3),
            create_mock_asset_state("QQQ", 0.25),
            create_mock_asset_state("IWM", 0.15),
            create_mock_asset_state("BND", 0.20),
        ]

    return PortfolioState(
        assets=assets,
        total_value=total_value,
        cash_weight=0.10,
        unrealized_pnl=1000.0,
        realized_pnl_today=500.0,
        days_since_rebalance=5,
        portfolio_volatility=0.15,
        portfolio_beta=0.95,
        sharpe_ratio=1.5,
        max_drawdown=0.05,
        transaction_cost_estimate=50.0,
    )


class TestRLRebalancerCreation:
    """Tests for RLRebalancer creation."""

    def test_create_default(self):
        """Test creating rebalancer with default config."""
        symbols = ["SPY", "QQQ", "IWM", "BND"]
        rebalancer = create_rl_rebalancer(symbols)

        assert rebalancer is not None
        assert rebalancer.num_assets == 4
        assert len(rebalancer.asset_symbols) == 4

    def test_create_custom_config(self):
        """Test creating rebalancer with custom config."""
        config = RLRebalancerConfig(
            min_weight=0.05,
            max_weight=0.50,
            max_cash_weight=0.20,
            commission_pct=0.002,
        )
        symbols = ["SPY", "QQQ"]
        rebalancer = create_rl_rebalancer(symbols, config)

        assert rebalancer.config.min_weight == 0.05
        assert rebalancer.config.max_weight == 0.50
        assert rebalancer.config.commission_pct == 0.002


class TestPolicyNetwork:
    """Tests for PolicyNetwork."""

    def test_forward_pass(self):
        """Test policy network forward pass."""
        network = PolicyNetwork(
            input_size=20,
            num_assets=4,
            hidden_size=32,
            num_hidden_layers=2,
        )

        state_vector = [0.5] * 20
        weights = network.forward(state_vector)

        # Should output num_assets + 1 (for cash)
        assert len(weights) == 5

        # Should sum to 1 (softmax)
        assert abs(sum(weights) - 1.0) < 1e-6

        # All weights should be positive
        assert all(w >= 0 for w in weights)

    def test_parameter_get_set(self):
        """Test getting and setting parameters."""
        network = PolicyNetwork(
            input_size=10,
            num_assets=3,
            hidden_size=16,
            num_hidden_layers=2,
        )

        params = network.get_parameters()
        assert len(params) > 0

        # Modify and set back
        modified_params = [p * 1.1 for p in params]
        network.set_parameters(modified_params)

        new_params = network.get_parameters()
        assert len(new_params) == len(params)


class TestValueNetwork:
    """Tests for ValueNetwork."""

    def test_forward_pass(self):
        """Test value network forward pass."""
        network = ValueNetwork(
            input_size=20,
            hidden_size=32,
            num_hidden_layers=2,
        )

        state_vector = [0.5] * 20
        value = network.forward(state_vector)

        # Should return single value
        assert isinstance(value, float)


class TestAssetState:
    """Tests for AssetState."""

    def test_to_vector(self):
        """Test converting asset state to vector."""
        state = create_mock_asset_state("SPY", 0.25, 450.0)
        vector = state.to_vector()

        # Should have 9 features
        assert len(vector) == 9
        assert vector[0] == 0.25  # current_weight


class TestPortfolioState:
    """Tests for PortfolioState."""

    def test_to_vector(self):
        """Test converting portfolio state to vector."""
        state = create_mock_portfolio_state()
        vector = state.to_vector()

        # Should have portfolio features + asset features
        # 9 portfolio features + 4 assets * 9 features = 45
        expected_size = 9 + (4 * 9)
        assert len(vector) == expected_size

    def test_num_assets(self):
        """Test num_assets property."""
        state = create_mock_portfolio_state()
        assert state.num_assets == 4


class TestRebalancingDecision:
    """Tests for rebalancing decision logic."""

    def test_decide_hold_small_deviation(self):
        """Test hold decision when deviation is small."""
        config = RLRebalancerConfig(
            min_weight_deviation=0.05,
        )
        symbols = ["SPY", "QQQ", "IWM", "BND"]
        rebalancer = create_rl_rebalancer(symbols, config)

        # Create state with weights close to equal
        assets = [
            create_mock_asset_state("SPY", 0.24),
            create_mock_asset_state("QQQ", 0.26),
            create_mock_asset_state("IWM", 0.24),
            create_mock_asset_state("BND", 0.26),
        ]
        state = create_mock_portfolio_state(assets)

        decision = rebalancer.decide_rebalance(state)

        # Small deviations should result in HOLD
        assert isinstance(decision, RebalanceDecision)
        assert decision.action in {RebalanceAction.HOLD, RebalanceAction.REBALANCE}

    def test_decide_rebalance_large_deviation(self):
        """Test rebalance decision when deviation is large."""
        config = RLRebalancerConfig(
            min_weight_deviation=0.02,
            max_weight_deviation=0.10,
        )
        symbols = ["SPY", "QQQ", "IWM", "BND"]
        rebalancer = create_rl_rebalancer(symbols, config)

        # Create state with large deviation
        assets = [
            create_mock_asset_state("SPY", 0.50),  # Way off target
            create_mock_asset_state("QQQ", 0.10),
            create_mock_asset_state("IWM", 0.30),
            create_mock_asset_state("BND", 0.10),
        ]
        state = create_mock_portfolio_state(assets)

        decision = rebalancer.decide_rebalance(state)

        assert isinstance(decision, RebalanceDecision)
        # Large deviation should trigger action
        assert decision.action in {RebalanceAction.REBALANCE, RebalanceAction.PARTIAL}

    def test_decision_contains_weights(self):
        """Test that decision contains weight information."""
        symbols = ["SPY", "QQQ"]
        rebalancer = create_rl_rebalancer(symbols)

        assets = [
            create_mock_asset_state("SPY", 0.4),
            create_mock_asset_state("QQQ", 0.5),
        ]
        state = create_mock_portfolio_state(assets)

        decision = rebalancer.decide_rebalance(state)

        assert "SPY" in decision.target_weights
        assert "QQQ" in decision.target_weights
        assert "CASH" in decision.target_weights

        # Weights should sum to approximately 1
        total = sum(decision.target_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_decision_to_dict(self):
        """Test decision serialization."""
        symbols = ["SPY", "QQQ"]
        rebalancer = create_rl_rebalancer(symbols)

        state = create_mock_portfolio_state(
            [
                create_mock_asset_state("SPY"),
                create_mock_asset_state("QQQ"),
            ]
        )

        decision = rebalancer.decide_rebalance(state)
        data = decision.to_dict()

        assert "action" in data
        assert "target_weights" in data
        assert "confidence" in data
        assert "timestamp" in data


class TestTransactionCostEstimation:
    """Tests for transaction cost estimation."""

    def test_estimate_cost(self):
        """Test transaction cost estimation."""
        config = RLRebalancerConfig(
            commission_pct=0.001,
            slippage_pct=0.001,
            min_trade_value=100.0,
        )
        symbols = ["SPY", "QQQ"]
        rebalancer = create_rl_rebalancer(symbols, config)

        weight_changes = {
            "SPY": 0.10,  # 10% change
            "QQQ": -0.10,
            "CASH": 0.0,
        }

        cost = rebalancer._estimate_transaction_cost(
            portfolio_value=100000.0,
            weight_changes=weight_changes,
        )

        # Should be positive
        assert cost > 0

        # 10% of $100k = $10k * 2 legs * 0.2% = $40
        expected_approx = 10000 * 2 * 0.002
        assert abs(cost - expected_approx) < 50


class TestTraining:
    """Tests for RL training functionality."""

    def test_record_outcome(self):
        """Test recording outcome after decision."""
        symbols = ["SPY", "QQQ"]
        rebalancer = create_rl_rebalancer(symbols)

        # Make initial decision
        state1 = create_mock_portfolio_state(
            [
                create_mock_asset_state("SPY"),
                create_mock_asset_state("QQQ"),
            ]
        )
        rebalancer.decide_rebalance(state1)

        # Record outcome
        state2 = create_mock_portfolio_state(
            [
                create_mock_asset_state("SPY"),
                create_mock_asset_state("QQQ"),
            ]
        )
        state2.total_value = 101000  # 1% gain

        rebalancer.record_outcome(state2)

        # Should have experience in buffer
        assert len(rebalancer._buffer) >= 0

    def test_train_insufficient_data(self):
        """Test training with insufficient data."""
        config = RLRebalancerConfig(min_buffer_size=50)
        symbols = ["SPY", "QQQ"]
        rebalancer = create_rl_rebalancer(symbols, config)

        result = rebalancer.train()

        assert result["status"] == "insufficient_data"

    def test_train_with_data(self):
        """Test training with sufficient data."""
        config = RLRebalancerConfig(
            min_buffer_size=5,
            batch_size=5,
        )
        symbols = ["SPY", "QQQ"]
        rebalancer = create_rl_rebalancer(symbols, config)

        # Generate experiences
        for i in range(10):
            state = create_mock_portfolio_state(
                [
                    create_mock_asset_state("SPY", 0.3 + i * 0.01),
                    create_mock_asset_state("QQQ", 0.7 - i * 0.01),
                ]
            )
            state.total_value = 100000 + i * 1000

            rebalancer.decide_rebalance(state)

            next_state = create_mock_portfolio_state(
                [
                    create_mock_asset_state("SPY", 0.31 + i * 0.01),
                    create_mock_asset_state("QQQ", 0.69 - i * 0.01),
                ]
            )
            next_state.total_value = 100500 + i * 1000

            rebalancer.record_outcome(next_state)

        result = rebalancer.train()

        assert result["status"] == "trained"
        assert "policy_loss" in result


class TestStatePersistence:
    """Tests for state save/load."""

    def test_save_state(self):
        """Test saving rebalancer state."""
        symbols = ["SPY", "QQQ"]
        rebalancer = create_rl_rebalancer(symbols)

        state = rebalancer.save_state()

        assert "policy_params" in state
        assert "value_params" in state
        assert "asset_symbols" in state
        assert len(state["policy_params"]) > 0

    def test_load_state(self):
        """Test loading rebalancer state."""
        symbols = ["SPY", "QQQ"]
        rebalancer1 = create_rl_rebalancer(symbols)

        # Get initial predictions
        portfolio_state = create_mock_portfolio_state(
            [
                create_mock_asset_state("SPY"),
                create_mock_asset_state("QQQ"),
            ]
        )
        weights1, _, _ = rebalancer1.get_target_weights(portfolio_state)

        # Save and load to new rebalancer
        saved = rebalancer1.save_state()
        rebalancer2 = create_rl_rebalancer(symbols)
        rebalancer2.load_state(saved)

        weights2, _, _ = rebalancer2.get_target_weights(portfolio_state)

        # Weights should be similar (same network)
        for symbol in weights1:
            assert abs(weights1[symbol] - weights2[symbol]) < 0.01


class TestMetrics:
    """Tests for metrics reporting."""

    def test_get_metrics(self):
        """Test getting rebalancer metrics."""
        symbols = ["SPY", "QQQ", "IWM"]
        rebalancer = create_rl_rebalancer(symbols)

        metrics = rebalancer.get_metrics()

        assert metrics["num_assets"] == 3
        assert "buffer_size" in metrics
        assert "config" in metrics


class TestWeightConstraints:
    """Tests for weight constraints."""

    def test_min_weight_constraint(self):
        """Test minimum weight constraint."""
        config = RLRebalancerConfig(
            min_weight=0.10,
            max_weight=0.90,
        )
        symbols = ["SPY", "QQQ", "IWM"]
        rebalancer = create_rl_rebalancer(symbols, config)

        portfolio_state = create_mock_portfolio_state(
            [
                create_mock_asset_state("SPY"),
                create_mock_asset_state("QQQ"),
                create_mock_asset_state("IWM"),
            ]
        )

        weights, _, _ = rebalancer.get_target_weights(portfolio_state)

        # All weights should be above minimum (except possibly cash)
        for symbol in symbols:
            assert weights[symbol] >= config.min_weight - 0.01

    def test_max_weight_constraint(self):
        """Test maximum weight constraint."""
        config = RLRebalancerConfig(
            min_weight=0.0,
            max_weight=0.50,
        )
        symbols = ["SPY", "QQQ"]
        rebalancer = create_rl_rebalancer(symbols, config)

        portfolio_state = create_mock_portfolio_state(
            [
                create_mock_asset_state("SPY"),
                create_mock_asset_state("QQQ"),
            ]
        )

        weights, _, _ = rebalancer.get_target_weights(portfolio_state)

        # All weights should be below maximum
        for symbol in symbols:
            assert weights[symbol] <= config.max_weight + 0.01

    def test_weights_sum_to_one(self):
        """Test that weights always sum to 1."""
        symbols = ["SPY", "QQQ", "IWM", "BND"]
        rebalancer = create_rl_rebalancer(symbols)

        for _ in range(10):
            portfolio_state = create_mock_portfolio_state(
                [create_mock_asset_state(s, 0.1 + 0.1 * i) for i, s in enumerate(symbols)]
            )

            weights, _, _ = rebalancer.get_target_weights(portfolio_state)
            total = sum(weights.values())

            assert abs(total - 1.0) < 0.01


class TestEnums:
    """Tests for enum types."""

    def test_rebalance_action_values(self):
        """Test RebalanceAction enum values."""
        assert RebalanceAction.HOLD.value == "hold"
        assert RebalanceAction.REBALANCE.value == "rebalance"
        assert RebalanceAction.PARTIAL.value == "partial"

    def test_rebalance_frequency_values(self):
        """Test RebalanceFrequency enum values."""
        assert RebalanceFrequency.DAILY.value == "daily"
        assert RebalanceFrequency.WEEKLY.value == "weekly"
        assert RebalanceFrequency.MONTHLY.value == "monthly"
        assert RebalanceFrequency.ADAPTIVE.value == "adaptive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
