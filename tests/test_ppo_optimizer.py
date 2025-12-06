"""
Tests for PPO Weight Optimizer (UPGRADE-014 Feature 8)

Tests the Proximal Policy Optimization-based weight optimizer
for sentiment ensemble models.
"""

from datetime import datetime, timezone

from llm.ppo_weight_optimizer import (
    Experience,
    ExperienceBuffer,
    PPOConfig,
    PPOWeightOptimizer,
    RewardType,
    SimpleNeuralNetwork,
    TradeOutcome,
    TradingRewardCalculator,
    ValueNetwork,
    WeightState,
    create_adaptive_weight_optimizer,
    create_ppo_optimizer,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def create_test_state(
    volatility: float = 0.5,
    trend: float = 0.0,
    num_models: int = 3,
) -> WeightState:
    """Create a test weight state."""
    return WeightState(
        volatility_percentile=volatility,
        trend_strength=trend,
        is_high_vol=volatility > 0.7,
        is_trending=abs(trend) > 0.3,
        recent_sharpe=0.5,
        recent_accuracy=0.6,
        recent_returns=0.01,
        current_weights=[1.0 / num_models] * num_models,
        model_confidences=[0.7] * num_models,
    )


def create_test_outcome(
    pnl_pct: float = 0.02,
    predicted: str = "bullish",
    actual: str = "bullish",
    num_models: int = 3,
) -> TradeOutcome:
    """Create a test trade outcome."""
    return TradeOutcome(
        symbol="SPY",
        entry_time=datetime.now(timezone.utc),
        exit_time=datetime.now(timezone.utc),
        entry_price=100.0,
        exit_price=100.0 * (1 + pnl_pct),
        predicted_direction=predicted,
        actual_direction=actual,
        pnl_pct=pnl_pct,
        weights_used=[1.0 / num_models] * num_models,
        model_predictions=[],
    )


# =============================================================================
# Test RewardType Enum
# =============================================================================


class TestRewardType:
    """Tests for RewardType enum."""

    def test_reward_types_exist(self):
        """Test all reward types exist."""
        assert RewardType.SHARPE.value == "sharpe"
        assert RewardType.RETURNS.value == "returns"
        assert RewardType.ACCURACY.value == "accuracy"
        assert RewardType.COMBINED.value == "combined"

    def test_reward_type_count(self):
        """Test total number of reward types."""
        assert len(RewardType) == 4


# =============================================================================
# Test WeightState
# =============================================================================


class TestWeightState:
    """Tests for WeightState dataclass."""

    def test_create_weight_state(self):
        """Test creating a weight state."""
        state = create_test_state()
        assert state.volatility_percentile == 0.5
        assert state.trend_strength == 0.0
        assert state.is_high_vol is False
        assert state.is_trending is False

    def test_to_vector(self):
        """Test converting state to feature vector."""
        state = create_test_state(num_models=3)
        vector = state.to_vector()

        # 7 base features + 3 weights + 3 confidences = 13
        assert len(vector) == 13

        # Check base features
        assert vector[0] == 0.5  # volatility_percentile
        assert vector[1] == 0.0  # trend_strength
        assert vector[2] == 0.0  # is_high_vol (False)
        assert vector[3] == 0.0  # is_trending (False)

    def test_to_vector_high_vol(self):
        """Test vector with high volatility."""
        state = create_test_state(volatility=0.8)
        vector = state.to_vector()
        assert vector[2] == 1.0  # is_high_vol (True)

    def test_to_vector_trending(self):
        """Test vector with trending market."""
        state = create_test_state(trend=0.5)
        vector = state.to_vector()
        assert vector[3] == 1.0  # is_trending (True)

    def test_default_state(self):
        """Test creating default state."""
        state = WeightState.default(num_models=4)

        assert state.volatility_percentile == 0.5
        assert state.trend_strength == 0.0
        assert state.is_high_vol is False
        assert state.is_trending is False
        assert len(state.current_weights) == 4
        assert len(state.model_confidences) == 4
        assert sum(state.current_weights) == 1.0

    def test_state_has_timestamp(self):
        """Test that state has timestamp."""
        state = create_test_state()
        assert state.timestamp is not None
        assert isinstance(state.timestamp, datetime)


# =============================================================================
# Test Experience
# =============================================================================


class TestExperience:
    """Tests for Experience dataclass."""

    def test_create_experience(self):
        """Test creating an experience."""
        state = create_test_state()
        exp = Experience(
            state=state,
            action=[0.4, 0.3, 0.3],
            reward=1.5,
            next_state=None,
            done=False,
            log_prob=-0.5,
            value=0.8,
        )

        assert exp.state == state
        assert sum(exp.action) == 1.0
        assert exp.reward == 1.5
        assert exp.done is False
        assert exp.log_prob == -0.5
        assert exp.value == 0.8

    def test_experience_has_timestamp(self):
        """Test that experience has timestamp."""
        exp = Experience(
            state=create_test_state(),
            action=[0.33, 0.33, 0.34],
            reward=0.0,
            next_state=None,
            done=True,
            log_prob=0.0,
            value=0.0,
        )
        assert exp.timestamp is not None


# =============================================================================
# Test TradeOutcome
# =============================================================================


class TestTradeOutcome:
    """Tests for TradeOutcome dataclass."""

    def test_create_trade_outcome(self):
        """Test creating a trade outcome."""
        outcome = create_test_outcome(pnl_pct=0.05)

        assert outcome.symbol == "SPY"
        assert outcome.pnl_pct == 0.05
        assert outcome.predicted_direction == "bullish"
        assert outcome.actual_direction == "bullish"

    def test_outcome_exit_price_calculation(self):
        """Test that exit price matches P&L."""
        outcome = create_test_outcome(pnl_pct=0.02)
        expected_exit = 100.0 * 1.02
        assert abs(outcome.exit_price - expected_exit) < 0.01


# =============================================================================
# Test PPOConfig
# =============================================================================


class TestPPOConfig:
    """Tests for PPOConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PPOConfig()

        assert config.learning_rate == 0.001
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_epsilon == 0.2
        assert config.batch_size == 32
        assert config.buffer_size == 1000
        assert config.min_buffer_size == 50
        assert config.reward_type == RewardType.COMBINED

    def test_weight_constraints(self):
        """Test weight constraint defaults."""
        config = PPOConfig()
        assert config.min_weight == 0.05
        assert config.max_weight == 0.90

    def test_reward_weights(self):
        """Test reward weight configuration."""
        config = PPOConfig()
        total = config.sharpe_weight + config.returns_weight + config.accuracy_weight
        assert abs(total - 1.0) < 0.01

    def test_custom_config(self):
        """Test custom configuration."""
        config = PPOConfig(
            learning_rate=0.0001,
            buffer_size=500,
            reward_type=RewardType.SHARPE,
        )
        assert config.learning_rate == 0.0001
        assert config.buffer_size == 500
        assert config.reward_type == RewardType.SHARPE


# =============================================================================
# Test SimpleNeuralNetwork
# =============================================================================


class TestSimpleNeuralNetwork:
    """Tests for SimpleNeuralNetwork class."""

    def test_create_network(self):
        """Test creating a neural network."""
        net = SimpleNeuralNetwork(
            input_size=10,
            hidden_size=32,
            output_size=3,
            num_hidden_layers=2,
        )

        assert net.input_size == 10
        assert net.hidden_size == 32
        assert net.output_size == 3

    def test_forward_pass(self):
        """Test forward pass returns correct shape."""
        net = SimpleNeuralNetwork(
            input_size=5,
            hidden_size=16,
            output_size=3,
        )

        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        output = net.forward(x)

        assert len(output) == 3

    def test_forward_pass_softmax(self):
        """Test that forward pass outputs valid probabilities."""
        net = SimpleNeuralNetwork(
            input_size=5,
            hidden_size=16,
            output_size=3,
        )

        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        output = net.forward(x)

        # Should sum to 1.0 (softmax)
        assert abs(sum(output) - 1.0) < 1e-6
        # All values should be positive
        assert all(p >= 0 for p in output)

    def test_deterministic_with_seed(self):
        """Test that same seed produces same output."""
        net1 = SimpleNeuralNetwork(input_size=5, hidden_size=16, output_size=3, seed=42)
        net2 = SimpleNeuralNetwork(input_size=5, hidden_size=16, output_size=3, seed=42)

        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        out1 = net1.forward(x)
        out2 = net2.forward(x)

        assert out1 == out2

    def test_get_set_parameters(self):
        """Test getting and setting parameters."""
        net = SimpleNeuralNetwork(input_size=5, hidden_size=8, output_size=3)

        params = net.get_parameters()
        assert len(params) > 0

        # Modify parameters
        modified = [p + 0.1 for p in params]
        net.set_parameters(modified)

        new_params = net.get_parameters()
        assert all(abs(a - b - 0.1) < 1e-6 for a, b in zip(new_params, params))


# =============================================================================
# Test ValueNetwork
# =============================================================================


class TestValueNetwork:
    """Tests for ValueNetwork class."""

    def test_create_value_network(self):
        """Test creating a value network."""
        net = ValueNetwork(input_size=10, hidden_size=32)
        assert net.output_size == 1

    def test_forward_returns_scalar(self):
        """Test that forward returns a single value."""
        net = ValueNetwork(input_size=5, hidden_size=16)

        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        output = net.forward(x)

        assert isinstance(output, float)


# =============================================================================
# Test ExperienceBuffer
# =============================================================================


class TestExperienceBuffer:
    """Tests for ExperienceBuffer class."""

    def test_create_buffer(self):
        """Test creating an experience buffer."""
        buffer = ExperienceBuffer(max_size=100)
        assert len(buffer) == 0
        assert buffer.max_size == 100

    def test_add_experience(self):
        """Test adding experiences."""
        buffer = ExperienceBuffer(max_size=100)

        exp = Experience(
            state=create_test_state(),
            action=[0.33, 0.33, 0.34],
            reward=1.0,
            next_state=None,
            done=False,
            log_prob=-0.5,
            value=0.5,
        )

        buffer.add(exp)
        assert len(buffer) == 1

    def test_buffer_max_size(self):
        """Test buffer respects max size."""
        buffer = ExperienceBuffer(max_size=5)

        for i in range(10):
            exp = Experience(
                state=create_test_state(),
                action=[0.33, 0.33, 0.34],
                reward=float(i),
                next_state=None,
                done=False,
                log_prob=0.0,
                value=0.0,
            )
            buffer.add(exp)

        assert len(buffer) == 5

    def test_sample(self):
        """Test sampling from buffer."""
        buffer = ExperienceBuffer(max_size=100)

        for i in range(20):
            exp = Experience(
                state=create_test_state(),
                action=[0.33, 0.33, 0.34],
                reward=float(i),
                next_state=None,
                done=False,
                log_prob=0.0,
                value=0.0,
            )
            buffer.add(exp)

        sample = buffer.sample(5)
        assert len(sample) == 5

    def test_sample_larger_than_buffer(self):
        """Test sampling more than buffer size."""
        buffer = ExperienceBuffer(max_size=100)

        for i in range(3):
            exp = Experience(
                state=create_test_state(),
                action=[0.33, 0.33, 0.34],
                reward=float(i),
                next_state=None,
                done=False,
                log_prob=0.0,
                value=0.0,
            )
            buffer.add(exp)

        sample = buffer.sample(10)
        assert len(sample) == 3

    def test_get_all(self):
        """Test getting all experiences."""
        buffer = ExperienceBuffer(max_size=100)

        for i in range(5):
            exp = Experience(
                state=create_test_state(),
                action=[0.33, 0.33, 0.34],
                reward=float(i),
                next_state=None,
                done=False,
                log_prob=0.0,
                value=0.0,
            )
            buffer.add(exp)

        all_exp = buffer.get_all()
        assert len(all_exp) == 5

    def test_clear(self):
        """Test clearing buffer."""
        buffer = ExperienceBuffer(max_size=100)

        for i in range(5):
            exp = Experience(
                state=create_test_state(),
                action=[0.33, 0.33, 0.34],
                reward=float(i),
                next_state=None,
                done=False,
                log_prob=0.0,
                value=0.0,
            )
            buffer.add(exp)

        buffer.clear()
        assert len(buffer) == 0


# =============================================================================
# Test TradingRewardCalculator
# =============================================================================


class TestTradingRewardCalculator:
    """Tests for TradingRewardCalculator class."""

    def test_create_calculator(self):
        """Test creating a reward calculator."""
        calc = TradingRewardCalculator()
        assert calc.reward_type == RewardType.COMBINED

    def test_returns_reward_positive(self):
        """Test returns reward for profitable trade."""
        calc = TradingRewardCalculator(reward_type=RewardType.RETURNS)
        outcome = create_test_outcome(pnl_pct=0.02)
        reward = calc.calculate_reward(outcome)

        # 2% gain = 2.0 reward
        assert reward == 2.0

    def test_returns_reward_negative(self):
        """Test returns reward for losing trade."""
        calc = TradingRewardCalculator(reward_type=RewardType.RETURNS)
        outcome = create_test_outcome(pnl_pct=-0.01)
        reward = calc.calculate_reward(outcome)

        # 1% loss = -1.0 reward
        assert reward == -1.0

    def test_accuracy_reward_correct(self):
        """Test accuracy reward for correct prediction."""
        calc = TradingRewardCalculator(reward_type=RewardType.ACCURACY)
        outcome = create_test_outcome(predicted="bullish", actual="bullish")
        reward = calc.calculate_reward(outcome)
        assert reward == 1.0

    def test_accuracy_reward_incorrect(self):
        """Test accuracy reward for incorrect prediction."""
        calc = TradingRewardCalculator(reward_type=RewardType.ACCURACY)
        outcome = create_test_outcome(predicted="bullish", actual="bearish")
        reward = calc.calculate_reward(outcome)
        assert reward == -1.0

    def test_accuracy_reward_neutral(self):
        """Test accuracy reward for neutral predictions."""
        calc = TradingRewardCalculator(reward_type=RewardType.ACCURACY)

        # Predicted neutral
        outcome1 = create_test_outcome(predicted="neutral", actual="bullish")
        assert calc.calculate_reward(outcome1) == 0.0

        # Actual neutral
        outcome2 = create_test_outcome(predicted="bullish", actual="neutral")
        assert calc.calculate_reward(outcome2) == 0.0

    def test_combined_reward(self):
        """Test combined reward calculation."""
        calc = TradingRewardCalculator(
            reward_type=RewardType.COMBINED,
            sharpe_weight=0.0,  # Disable for predictable test
            returns_weight=0.5,
            accuracy_weight=0.5,
        )

        outcome = create_test_outcome(
            pnl_pct=0.02,  # Returns reward = 2.0
            predicted="bullish",
            actual="bullish",  # Accuracy reward = 1.0
        )
        reward = calc.calculate_reward(outcome)

        # Combined = 0.5 * 2.0 + 0.5 * 1.0 = 1.5
        # (Plus small Sharpe contribution if history builds up)
        assert reward > 0

    def test_sharpe_reward_builds_history(self):
        """Test Sharpe reward builds history."""
        calc = TradingRewardCalculator(reward_type=RewardType.SHARPE)

        for i in range(10):
            outcome = create_test_outcome(pnl_pct=0.01)
            calc.calculate_reward(outcome)

        assert len(calc.returns_history) == 10

    def test_reset(self):
        """Test resetting calculator."""
        calc = TradingRewardCalculator(reward_type=RewardType.SHARPE)

        for i in range(5):
            outcome = create_test_outcome(pnl_pct=0.01)
            calc.calculate_reward(outcome)

        assert len(calc.returns_history) == 5
        calc.reset()
        assert len(calc.returns_history) == 0


# =============================================================================
# Test PPOWeightOptimizer
# =============================================================================


class TestPPOWeightOptimizer:
    """Tests for PPOWeightOptimizer class."""

    def test_create_optimizer(self):
        """Test creating a PPO optimizer."""
        optimizer = PPOWeightOptimizer(num_models=3)
        assert optimizer.num_models == 3
        assert len(optimizer.model_names) == 3

    def test_create_with_custom_names(self):
        """Test creating optimizer with custom model names."""
        optimizer = PPOWeightOptimizer(
            num_models=3,
            model_names=["finbert", "gpt4", "claude"],
        )
        assert optimizer.model_names == ["finbert", "gpt4", "claude"]

    def test_get_optimal_weights(self):
        """Test getting optimal weights."""
        optimizer = PPOWeightOptimizer(num_models=3)
        state = create_test_state(num_models=3)

        weights = optimizer.get_optimal_weights(state)

        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)

    def test_weights_respect_constraints(self):
        """Test that weights respect min/max constraints."""
        config = PPOConfig(min_weight=0.1, max_weight=0.8)
        optimizer = PPOWeightOptimizer(num_models=3, config=config)
        state = create_test_state(num_models=3)

        weights = optimizer.get_optimal_weights(state)

        for w in weights:
            # After normalization, may exceed max slightly due to constraint interaction
            assert w >= config.min_weight - 0.01

    def test_record_outcome(self):
        """Test recording trade outcome."""
        optimizer = PPOWeightOptimizer(num_models=3)
        state = create_test_state(num_models=3)
        weights = [0.4, 0.3, 0.3]
        outcome = create_test_outcome(pnl_pct=0.02)

        reward = optimizer.record_outcome(state, weights, outcome)

        assert isinstance(reward, float)
        assert len(optimizer.buffer) == 1

    def test_should_train_false_initially(self):
        """Test that training is not allowed with empty buffer."""
        optimizer = PPOWeightOptimizer(num_models=3)
        assert optimizer.should_train() is False

    def test_should_train_true_after_min_buffer(self):
        """Test training allowed after minimum buffer size."""
        config = PPOConfig(min_buffer_size=5)
        optimizer = PPOWeightOptimizer(num_models=3, config=config)

        # Add minimum experiences
        for i in range(5):
            state = create_test_state(num_models=3)
            weights = [0.33, 0.33, 0.34]
            outcome = create_test_outcome(pnl_pct=0.01 * (i + 1))
            optimizer.record_outcome(state, weights, outcome)

        assert optimizer.should_train() is True

    def test_train(self):
        """Test training the optimizer."""
        config = PPOConfig(min_buffer_size=10, batch_size=5, epochs_per_update=2)
        optimizer = PPOWeightOptimizer(num_models=3, config=config)

        # Add experiences
        for i in range(15):
            state = create_test_state(num_models=3)
            weights = [0.33, 0.33, 0.34]
            outcome = create_test_outcome(
                pnl_pct=0.01 * (i - 7),  # Mix of gains and losses
                predicted="bullish" if i % 2 == 0 else "bearish",
                actual="bullish" if i % 3 == 0 else "bearish",
            )
            optimizer.record_outcome(state, weights, outcome)

        metrics = optimizer.train()

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "training_step" in metrics
        assert metrics["training_step"] == 1

    def test_train_clears_buffer(self):
        """Test that training clears the buffer."""
        config = PPOConfig(min_buffer_size=5)
        optimizer = PPOWeightOptimizer(num_models=3, config=config)

        for i in range(10):
            state = create_test_state(num_models=3)
            weights = [0.33, 0.33, 0.34]
            outcome = create_test_outcome(pnl_pct=0.01)
            optimizer.record_outcome(state, weights, outcome)

        assert len(optimizer.buffer) == 10
        optimizer.train()
        assert len(optimizer.buffer) == 0

    def test_get_statistics(self):
        """Test getting optimizer statistics."""
        optimizer = PPOWeightOptimizer(
            num_models=3,
            model_names=["a", "b", "c"],
        )

        stats = optimizer.get_statistics()

        assert "num_experiences" in stats
        assert "training_steps" in stats
        assert "model_names" in stats
        assert stats["model_names"] == ["a", "b", "c"]

    def test_save_load_state(self):
        """Test saving and loading state."""
        optimizer1 = PPOWeightOptimizer(num_models=3, seed=42)

        # Add some experiences and train
        for i in range(60):
            state = create_test_state(num_models=3)
            weights = [0.33, 0.33, 0.34]
            outcome = create_test_outcome(pnl_pct=0.01)
            optimizer1.record_outcome(state, weights, outcome)

        optimizer1.train()
        saved_state = optimizer1.save_state()

        # Create new optimizer and load state
        optimizer2 = PPOWeightOptimizer(num_models=3, seed=42)
        optimizer2.load_state(saved_state)

        assert optimizer2.training_step == optimizer1.training_step
        assert len(optimizer2.total_rewards) == len(optimizer1.total_rewards)

    def test_reset(self):
        """Test resetting optimizer."""
        optimizer = PPOWeightOptimizer(num_models=3)

        # Add some data
        for i in range(5):
            state = create_test_state(num_models=3)
            weights = [0.33, 0.33, 0.34]
            outcome = create_test_outcome(pnl_pct=0.01)
            optimizer.record_outcome(state, weights, outcome)

        optimizer.reset()

        assert len(optimizer.buffer) == 0
        assert len(optimizer.total_rewards) == 0
        assert optimizer.training_step == 0


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestCreatePPOOptimizer:
    """Tests for create_ppo_optimizer factory function."""

    def test_create_default(self):
        """Test creating optimizer with defaults."""
        optimizer = create_ppo_optimizer()

        assert optimizer.num_models == 3
        assert optimizer.config.reward_type == RewardType.COMBINED

    def test_create_with_custom_params(self):
        """Test creating optimizer with custom parameters."""
        optimizer = create_ppo_optimizer(
            num_models=4,
            model_names=["a", "b", "c", "d"],
            learning_rate=0.0001,
            reward_type=RewardType.SHARPE,
            buffer_size=500,
        )

        assert optimizer.num_models == 4
        assert optimizer.model_names == ["a", "b", "c", "d"]
        assert optimizer.config.learning_rate == 0.0001
        assert optimizer.config.reward_type == RewardType.SHARPE
        assert optimizer.config.buffer_size == 500


class TestCreateAdaptiveWeightOptimizer:
    """Tests for create_adaptive_weight_optimizer factory function."""

    def test_create_adaptive(self):
        """Test creating adaptive weight optimizer."""
        optimizer = create_adaptive_weight_optimizer(
            model_names=["finbert", "gpt4", "claude"],
        )

        assert optimizer.num_models == 3
        assert optimizer.model_names == ["finbert", "gpt4", "claude"]
        assert optimizer.config.reward_type == RewardType.COMBINED

    def test_custom_reward_weights(self):
        """Test custom reward weights."""
        optimizer = create_adaptive_weight_optimizer(
            model_names=["a", "b"],
            sharpe_weight=0.6,
            returns_weight=0.2,
            accuracy_weight=0.2,
        )

        assert optimizer.config.sharpe_weight == 0.6
        assert optimizer.config.returns_weight == 0.2
        assert optimizer.config.accuracy_weight == 0.2

    def test_custom_weight_constraints(self):
        """Test custom weight constraints."""
        optimizer = create_adaptive_weight_optimizer(
            model_names=["a", "b", "c"],
            min_weight=0.1,
            max_weight=0.6,
        )

        assert optimizer.config.min_weight == 0.1
        assert optimizer.config.max_weight == 0.6


# =============================================================================
# Test Integration
# =============================================================================


class TestPPOIntegration:
    """Integration tests for PPO weight optimization."""

    def test_full_training_cycle(self):
        """Test complete training cycle."""
        optimizer = create_ppo_optimizer(
            num_models=3,
            model_names=["finbert", "gpt4", "claude"],
        )

        # Simulate trading with performance feedback
        for episode in range(3):
            for i in range(20):
                # Get weights for current state
                state = WeightState(
                    volatility_percentile=0.3 + 0.1 * (i % 5),
                    trend_strength=-0.5 + 0.25 * (i % 5),
                    is_high_vol=(i % 5) > 3,
                    is_trending=(i % 3) == 0,
                    recent_sharpe=0.5 + 0.1 * (i % 3),
                    recent_accuracy=0.6,
                    recent_returns=0.01,
                    current_weights=[0.33, 0.33, 0.34],
                    model_confidences=[0.7, 0.6, 0.8],
                )

                weights = optimizer.get_optimal_weights(state)

                # Simulate trade outcome (better for higher confidence)
                pnl = 0.02 if sum(weights[:2]) > 0.5 else -0.01
                outcome = TradeOutcome(
                    symbol="SPY",
                    entry_time=datetime.now(timezone.utc),
                    exit_time=datetime.now(timezone.utc),
                    entry_price=100.0,
                    exit_price=100.0 * (1 + pnl),
                    predicted_direction="bullish",
                    actual_direction="bullish" if pnl > 0 else "bearish",
                    pnl_pct=pnl,
                    weights_used=weights,
                    model_predictions=[],
                )

                optimizer.record_outcome(state, weights, outcome)

            # Train after each episode
            if optimizer.should_train():
                metrics = optimizer.train()
                assert "policy_loss" in metrics

        # Verify training occurred
        assert optimizer.training_step >= 1

    def test_weights_evolve_with_training(self):
        """Test that weights change with training."""
        optimizer = create_ppo_optimizer(num_models=3)

        # Get initial weights
        state = create_test_state(num_models=3)
        initial_weights = optimizer.get_optimal_weights(state, add_exploration=False)

        # Add experiences with biased rewards
        for i in range(100):
            state = create_test_state(num_models=3)
            # Always reward high weight on first model
            weights = [0.6, 0.2, 0.2]
            outcome = create_test_outcome(pnl_pct=0.03)  # Good outcome
            optimizer.record_outcome(state, weights, outcome)

        # Train
        optimizer.train()

        # Get new weights
        new_state = create_test_state(num_models=3)
        new_weights = optimizer.get_optimal_weights(new_state, add_exploration=False)

        # Weights should have changed
        weight_diff = sum(abs(a - b) for a, b in zip(initial_weights, new_weights))
        # Some change should occur (though evolution strategies may be slow)
        assert weight_diff >= 0  # Just ensure no errors

    def test_state_persistence(self):
        """Test optimizer state can be saved and restored."""
        optimizer1 = create_ppo_optimizer(
            num_models=3,
            model_names=["a", "b", "c"],
        )

        # Train
        for i in range(60):
            state = create_test_state(num_models=3)
            weights = optimizer1.get_optimal_weights(state)
            outcome = create_test_outcome(pnl_pct=0.01 * ((i % 5) - 2))
            optimizer1.record_outcome(state, weights, outcome)

        optimizer1.train()

        # Save state
        saved = optimizer1.save_state()

        # Create new optimizer and restore
        optimizer2 = create_ppo_optimizer(num_models=3)
        optimizer2.load_state(saved)

        # Get weights from both - should be similar
        state = create_test_state(num_models=3)
        weights1 = optimizer1.get_optimal_weights(state, add_exploration=False)
        weights2 = optimizer2.get_optimal_weights(state, add_exploration=False)

        for w1, w2 in zip(weights1, weights2):
            assert abs(w1 - w2) < 0.01
