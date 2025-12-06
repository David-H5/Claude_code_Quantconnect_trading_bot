"""
PPO-Optimized Weight Optimizer for Sentiment Ensemble

UPGRADE-014 Feature 8: PPO-Optimized Weighting

Implements Proximal Policy Optimization (PPO) to dynamically optimize
ensemble weights based on trading performance. This lightweight implementation
uses numpy instead of PyTorch for minimal dependencies.

Research Source: Adaptive Alpha Weighting with PPO (arXiv Sep 2025)
- PPO-optimized alpha weighting outperforms static weights
- Dynamic weighting based on performance improves returns
- Continuous learning from trading outcomes

Key Features:
- Lightweight numpy-based PPO implementation
- Trading-specific reward functions (Sharpe, returns, accuracy)
- Experience buffer for batch training
- Market regime-aware state representation
- Compatible with existing SentimentFilter and ensemble systems

QuantConnect Compatible: Yes
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# Use numpy if available, fallback to pure Python
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class RewardType(Enum):
    """Types of rewards for weight optimization."""

    SHARPE = "sharpe"  # Sharpe ratio based reward
    RETURNS = "returns"  # Raw returns based reward
    ACCURACY = "accuracy"  # Prediction accuracy based reward
    COMBINED = "combined"  # Weighted combination of all


@dataclass
class WeightState:
    """State representation for weight optimization."""

    # Market regime indicators
    volatility_percentile: float  # 0.0 to 1.0
    trend_strength: float  # -1.0 to 1.0
    is_high_vol: bool
    is_trending: bool

    # Recent performance metrics
    recent_sharpe: float
    recent_accuracy: float
    recent_returns: float

    # Current weights (to enable incremental updates)
    current_weights: list[float]

    # Model-specific metrics (e.g., recent confidence per model)
    model_confidences: list[float]

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_vector(self) -> list[float]:
        """Convert state to feature vector."""
        features = [
            self.volatility_percentile,
            self.trend_strength,
            1.0 if self.is_high_vol else 0.0,
            1.0 if self.is_trending else 0.0,
            self.recent_sharpe,
            self.recent_accuracy,
            self.recent_returns,
        ]
        features.extend(self.current_weights)
        features.extend(self.model_confidences)
        return features

    @classmethod
    def default(cls, num_models: int = 3) -> "WeightState":
        """Create default state."""
        return cls(
            volatility_percentile=0.5,
            trend_strength=0.0,
            is_high_vol=False,
            is_trending=False,
            recent_sharpe=0.0,
            recent_accuracy=0.5,
            recent_returns=0.0,
            current_weights=[1.0 / num_models] * num_models,
            model_confidences=[0.5] * num_models,
        )


@dataclass
class Experience:
    """Single experience tuple for training."""

    state: WeightState
    action: list[float]  # Chosen weights
    reward: float
    next_state: WeightState | None
    done: bool
    log_prob: float  # Log probability of action under policy
    value: float  # Estimated value of state
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TradeOutcome:
    """Outcome of a trade for reward calculation."""

    symbol: str
    entry_time: datetime
    exit_time: datetime | None
    entry_price: float
    exit_price: float | None
    predicted_direction: str  # "bullish", "bearish", "neutral"
    actual_direction: str | None  # After trade
    pnl_pct: float | None  # P&L percentage
    weights_used: list[float]
    model_predictions: list[dict[str, Any]]
    holding_period_hours: float | None = None


@dataclass
class PPOConfig:
    """Configuration for PPO optimizer."""

    # PPO hyperparameters
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clipping
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01  # Encourage exploration

    # Training parameters
    batch_size: int = 32
    epochs_per_update: int = 4
    buffer_size: int = 1000
    min_buffer_size: int = 50  # Min experiences before training

    # Weight constraints
    min_weight: float = 0.05  # Minimum weight per model
    max_weight: float = 0.90  # Maximum weight per model

    # Reward configuration
    reward_type: RewardType = RewardType.COMBINED
    sharpe_weight: float = 0.4
    returns_weight: float = 0.3
    accuracy_weight: float = 0.3

    # Network architecture
    hidden_size: int = 64
    num_hidden_layers: int = 2


class SimpleNeuralNetwork:
    """
    Simple feedforward neural network using numpy.

    This is a lightweight implementation for environments without PyTorch.
    Uses tanh activation and softmax output for weight probabilities.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_hidden_layers: int = 2,
        seed: int = 42,
    ):
        """Initialize network with random weights."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers

        # Set seed for reproducibility
        random.seed(seed)

        # Initialize weights (Xavier initialization)
        self.weights = []
        self.biases = []

        # Input -> First hidden
        self.weights.append(self._init_weights(input_size, hidden_size))
        self.biases.append(self._init_bias(hidden_size))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.weights.append(self._init_weights(hidden_size, hidden_size))
            self.biases.append(self._init_bias(hidden_size))

        # Last hidden -> Output
        self.weights.append(self._init_weights(hidden_size, output_size))
        self.biases.append(self._init_bias(output_size))

    def _init_weights(self, fan_in: int, fan_out: int) -> list[list[float]]:
        """Xavier initialization."""
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return [[random.gauss(0, std) for _ in range(fan_out)] for _ in range(fan_in)]

    def _init_bias(self, size: int) -> list[float]:
        """Initialize biases to zero."""
        return [0.0] * size

    def _tanh(self, x: float) -> float:
        """Tanh activation."""
        try:
            return math.tanh(x)
        except OverflowError:
            return 1.0 if x > 0 else -1.0

    def _softmax(self, x: list[float]) -> list[float]:
        """Softmax for output probabilities."""
        max_x = max(x)
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp = sum(exp_x)
        return [e / sum_exp for e in exp_x]

    def _matmul(
        self,
        x: list[float],
        weights: list[list[float]],
        bias: list[float],
    ) -> list[float]:
        """Matrix multiplication: x @ weights + bias."""
        result = []
        for j in range(len(weights[0])):
            val = bias[j]
            for i in range(len(x)):
                val += x[i] * weights[i][j]
            result.append(val)
        return result

    def forward(self, x: list[float]) -> list[float]:
        """Forward pass through the network."""
        current = x

        # Hidden layers with tanh
        for i in range(len(self.weights) - 1):
            current = self._matmul(current, self.weights[i], self.biases[i])
            current = [self._tanh(v) for v in current]

        # Output layer with softmax (for weight probabilities)
        output = self._matmul(current, self.weights[-1], self.biases[-1])
        return self._softmax(output)

    def get_parameters(self) -> list[float]:
        """Flatten all parameters into a single list."""
        params = []
        for w in self.weights:
            for row in w:
                params.extend(row)
        for b in self.biases:
            params.extend(b)
        return params

    def set_parameters(self, params: list[float]) -> None:
        """Set parameters from flattened list."""
        idx = 0
        for layer_idx, w in enumerate(self.weights):
            for row_idx in range(len(w)):
                for col_idx in range(len(w[0])):
                    self.weights[layer_idx][row_idx][col_idx] = params[idx]
                    idx += 1
        for layer_idx, b in enumerate(self.biases):
            for i in range(len(b)):
                self.biases[layer_idx][i] = params[idx]
                idx += 1


class ValueNetwork(SimpleNeuralNetwork):
    """
    Value function network for PPO.

    Estimates the value of a state (expected future reward).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
    ):
        """Initialize value network (single output)."""
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=1,
            num_hidden_layers=num_hidden_layers,
        )

    def forward(self, x: list[float]) -> float:
        """Forward pass returning single value."""
        current = x

        # Hidden layers with tanh
        for i in range(len(self.weights) - 1):
            current = self._matmul(current, self.weights[i], self.biases[i])
            current = [self._tanh(v) for v in current]

        # Output layer (no activation for value)
        output = self._matmul(current, self.weights[-1], self.biases[-1])
        return output[0]


class ExperienceBuffer:
    """
    Experience replay buffer for PPO training.

    Stores (state, action, reward, next_state, done) tuples
    for batch training.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize buffer."""
        self.max_size = max_size
        self.buffer: list[Experience] = []
        self._position = 0

    def add(self, experience: Experience) -> None:
        """Add experience to buffer."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self._position] = experience
        self._position = (self._position + 1) % self.max_size

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample random batch from buffer."""
        if batch_size >= len(self.buffer):
            return self.buffer.copy()
        return random.sample(self.buffer, batch_size)

    def get_all(self) -> list[Experience]:
        """Get all experiences."""
        return self.buffer.copy()

    def clear(self) -> None:
        """Clear buffer."""
        self.buffer = []
        self._position = 0

    def __len__(self) -> int:
        return len(self.buffer)


class TradingRewardCalculator:
    """
    Calculate rewards from trading outcomes.

    Supports multiple reward types:
    - Sharpe ratio based (risk-adjusted)
    - Raw returns
    - Prediction accuracy
    - Combined (weighted average)
    """

    def __init__(
        self,
        reward_type: RewardType = RewardType.COMBINED,
        sharpe_weight: float = 0.4,
        returns_weight: float = 0.3,
        accuracy_weight: float = 0.3,
        risk_free_rate: float = 0.0,
    ):
        """Initialize reward calculator."""
        self.reward_type = reward_type
        self.sharpe_weight = sharpe_weight
        self.returns_weight = returns_weight
        self.accuracy_weight = accuracy_weight
        self.risk_free_rate = risk_free_rate

        # Track outcomes for Sharpe calculation
        self.returns_history: list[float] = []
        self.max_history = 50

    def calculate_reward(
        self,
        outcome: TradeOutcome,
        outcomes_batch: list[TradeOutcome] | None = None,
    ) -> float:
        """
        Calculate reward from trade outcome.

        Args:
            outcome: Single trade outcome
            outcomes_batch: Optional batch for Sharpe calculation

        Returns:
            Calculated reward
        """
        if self.reward_type == RewardType.SHARPE:
            return self._sharpe_reward(outcome, outcomes_batch)
        elif self.reward_type == RewardType.RETURNS:
            return self._returns_reward(outcome)
        elif self.reward_type == RewardType.ACCURACY:
            return self._accuracy_reward(outcome)
        else:  # COMBINED
            return self._combined_reward(outcome, outcomes_batch)

    def _returns_reward(self, outcome: TradeOutcome) -> float:
        """Reward based on returns."""
        if outcome.pnl_pct is None:
            return 0.0

        # Scale returns to reasonable reward range
        # 1% gain = 1.0 reward, 1% loss = -1.0 reward
        return outcome.pnl_pct * 100

    def _accuracy_reward(self, outcome: TradeOutcome) -> float:
        """Reward based on prediction accuracy."""
        if outcome.actual_direction is None:
            return 0.0

        # Correct prediction = +1, incorrect = -1
        predicted = outcome.predicted_direction.lower()
        actual = outcome.actual_direction.lower()

        if predicted == actual:
            return 1.0
        elif predicted == "neutral" or actual == "neutral":
            return 0.0
        else:
            return -1.0

    def _sharpe_reward(
        self,
        outcome: TradeOutcome,
        outcomes_batch: list[TradeOutcome] | None = None,
    ) -> float:
        """Reward based on Sharpe ratio contribution."""
        if outcome.pnl_pct is None:
            return 0.0

        # Add to history
        self.returns_history.append(outcome.pnl_pct)
        if len(self.returns_history) > self.max_history:
            self.returns_history.pop(0)

        # Calculate Sharpe of recent returns
        if len(self.returns_history) < 5:
            return self._returns_reward(outcome)

        mean_return = sum(self.returns_history) / len(self.returns_history)
        variance = sum((r - mean_return) ** 2 for r in self.returns_history) / len(self.returns_history)
        std_dev = math.sqrt(variance) if variance > 0 else 0.001

        sharpe = (mean_return - self.risk_free_rate) / std_dev

        # Scale to reward range
        return sharpe * 2  # Sharpe of 1 = reward of 2

    def _combined_reward(
        self,
        outcome: TradeOutcome,
        outcomes_batch: list[TradeOutcome] | None = None,
    ) -> float:
        """Combined reward from all sources."""
        returns_r = self._returns_reward(outcome)
        accuracy_r = self._accuracy_reward(outcome)
        sharpe_r = self._sharpe_reward(outcome, outcomes_batch)

        return self.returns_weight * returns_r + self.accuracy_weight * accuracy_r + self.sharpe_weight * sharpe_r

    def reset(self) -> None:
        """Reset history."""
        self.returns_history = []


class PPOWeightOptimizer:
    """
    PPO-based weight optimizer for sentiment ensemble.

    Uses Proximal Policy Optimization to learn optimal weights
    for combining multiple sentiment models based on trading
    performance.

    Usage:
        optimizer = PPOWeightOptimizer(num_models=3)

        # Get optimized weights for current state
        state = WeightState(
            volatility_percentile=0.7,
            trend_strength=0.3,
            ...
        )
        weights = optimizer.get_optimal_weights(state)

        # After trade completes, record outcome
        outcome = TradeOutcome(...)
        optimizer.record_outcome(state, weights, outcome)

        # Periodically train on collected experiences
        if optimizer.should_train():
            loss = optimizer.train()

    Based on: "Adaptive Alpha Weighting with PPO" (arXiv Sep 2025)
    """

    def __init__(
        self,
        num_models: int = 3,
        model_names: list[str] | None = None,
        config: PPOConfig | None = None,
        seed: int = 42,
    ):
        """
        Initialize PPO weight optimizer.

        Args:
            num_models: Number of models in ensemble
            model_names: Names of models (for logging)
            config: PPO configuration
            seed: Random seed
        """
        self.num_models = num_models
        self.model_names = model_names or [f"model_{i}" for i in range(num_models)]
        self.config = config or PPOConfig()
        self.seed = seed

        random.seed(seed)

        # Calculate state size
        # 7 base features + num_models (current weights) + num_models (confidences)
        self.state_size = 7 + num_models * 2

        # Initialize networks
        self.policy_network = SimpleNeuralNetwork(
            input_size=self.state_size,
            hidden_size=self.config.hidden_size,
            output_size=num_models,
            num_hidden_layers=self.config.num_hidden_layers,
            seed=seed,
        )

        self.value_network = ValueNetwork(
            input_size=self.state_size,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
        )

        # Initialize buffer and reward calculator
        self.buffer = ExperienceBuffer(max_size=self.config.buffer_size)
        self.reward_calculator = TradingRewardCalculator(
            reward_type=self.config.reward_type,
            sharpe_weight=self.config.sharpe_weight,
            returns_weight=self.config.returns_weight,
            accuracy_weight=self.config.accuracy_weight,
        )

        # Training state
        self.training_step = 0
        self.total_rewards: list[float] = []
        self.avg_weights_history: list[list[float]] = []

        # Current episode state
        self._current_state: WeightState | None = None
        self._current_action: list[float] | None = None

    def get_optimal_weights(
        self,
        state: WeightState,
        add_exploration: bool = True,
    ) -> list[float]:
        """
        Get optimal weights for current state.

        Args:
            state: Current market/model state
            add_exploration: Whether to add exploration noise

        Returns:
            List of weights summing to 1.0
        """
        # Convert state to feature vector
        state_vector = state.to_vector()

        # Get raw probabilities from policy network
        raw_weights = self.policy_network.forward(state_vector)

        # Add exploration noise if training
        if add_exploration and len(self.buffer) < self.config.buffer_size:
            exploration_scale = 0.1 * (1 - len(self.buffer) / self.config.buffer_size)
            noise = [random.gauss(0, exploration_scale) for _ in raw_weights]
            raw_weights = [w + n for w, n in zip(raw_weights, noise)]

        # Apply weight constraints
        weights = self._apply_constraints(raw_weights)

        # Store for later
        self._current_state = state
        self._current_action = weights

        return weights

    def _apply_constraints(self, raw_weights: list[float]) -> list[float]:
        """Apply min/max constraints and normalize."""
        # Clip to constraints
        weights = [max(self.config.min_weight, min(self.config.max_weight, w)) for w in raw_weights]

        # Normalize to sum to 1
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / self.num_models] * self.num_models

        return weights

    def record_outcome(
        self,
        state: WeightState,
        weights_used: list[float],
        outcome: TradeOutcome,
        next_state: WeightState | None = None,
        done: bool = False,
    ) -> float:
        """
        Record trade outcome for training.

        Args:
            state: State when weights were chosen
            weights_used: Weights that were used
            outcome: Trade outcome
            next_state: State after trade (optional)
            done: Whether episode is done

        Returns:
            Calculated reward
        """
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(outcome)

        # Get log probability and value estimate
        state_vector = state.to_vector()
        probs = self.policy_network.forward(state_vector)
        log_prob = sum(w * math.log(max(p, 1e-8)) for w, p in zip(weights_used, probs))
        value = self.value_network.forward(state_vector)

        # Create and store experience
        experience = Experience(
            state=state,
            action=weights_used,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value,
        )
        self.buffer.add(experience)

        # Track rewards
        self.total_rewards.append(reward)

        return reward

    def should_train(self) -> bool:
        """Check if we have enough experiences to train."""
        return len(self.buffer) >= self.config.min_buffer_size

    def train(self) -> dict[str, float]:
        """
        Train policy and value networks on collected experiences.

        Returns:
            Dictionary of training metrics
        """
        if not self.should_train():
            return {"error": "Not enough experiences"}

        experiences = self.buffer.get_all()

        # Calculate returns and advantages using GAE
        returns, advantages = self._compute_gae(experiences)

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        # Multiple epochs over the data
        for _ in range(self.config.epochs_per_update):
            # Sample batch
            batch_indices = list(range(len(experiences)))
            random.shuffle(batch_indices)

            for i in range(0, len(batch_indices), self.config.batch_size):
                batch_idx = batch_indices[i : i + self.config.batch_size]

                # Compute losses and update
                policy_loss, value_loss, entropy = self._update_step(
                    [experiences[j] for j in batch_idx],
                    [returns[j] for j in batch_idx],
                    [advantages[j] for j in batch_idx],
                )

                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy
                num_updates += 1

        self.training_step += 1

        # Track average weights
        avg_weights = [0.0] * self.num_models
        for exp in experiences:
            for i, w in enumerate(exp.action):
                avg_weights[i] += w
        avg_weights = [w / len(experiences) for w in avg_weights]
        self.avg_weights_history.append(avg_weights)

        # Clear buffer after training
        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "avg_reward": sum(self.total_rewards[-100:]) / max(len(self.total_rewards[-100:]), 1),
            "training_step": self.training_step,
            "avg_weights": avg_weights,
        }

    def _compute_gae(
        self,
        experiences: list[Experience],
    ) -> tuple[list[float], list[float]]:
        """Compute Generalized Advantage Estimation."""
        returns = []
        advantages = []

        next_value = 0.0
        next_advantage = 0.0

        # Process in reverse order
        for exp in reversed(experiences):
            if exp.done:
                next_value = 0.0
                next_advantage = 0.0

            # Get next state value
            if exp.next_state is not None:
                next_value = self.value_network.forward(exp.next_state.to_vector())

            # TD error
            td_error = exp.reward + self.config.gamma * next_value * (0.0 if exp.done else 1.0) - exp.value

            # GAE advantage
            advantage = td_error + self.config.gamma * self.config.gae_lambda * next_advantage * (
                0.0 if exp.done else 1.0
            )

            # Return
            ret = advantage + exp.value

            returns.insert(0, ret)
            advantages.insert(0, advantage)

            next_advantage = advantage
            next_value = exp.value

        # Normalize advantages
        mean_adv = sum(advantages) / len(advantages)
        std_adv = math.sqrt(sum((a - mean_adv) ** 2 for a in advantages) / len(advantages))
        if std_adv > 1e-8:
            advantages = [(a - mean_adv) / std_adv for a in advantages]

        return returns, advantages

    def _update_step(
        self,
        batch: list[Experience],
        returns: list[float],
        advantages: list[float],
    ) -> tuple[float, float, float]:
        """
        Perform single PPO update step.

        This is a simplified gradient-free update using evolution strategies
        since we don't have automatic differentiation.
        """
        policy_loss = 0.0
        value_loss = 0.0
        entropy = 0.0

        for exp, ret, adv in zip(batch, returns, advantages):
            state_vector = exp.state.to_vector()

            # Current policy probabilities
            probs = self.policy_network.forward(state_vector)

            # New log probability
            new_log_prob = sum(w * math.log(max(p, 1e-8)) for w, p in zip(exp.action, probs))

            # Probability ratio
            ratio = math.exp(new_log_prob - exp.log_prob)

            # Clipped surrogate objective
            surr1 = ratio * adv
            surr2 = max(min(ratio, 1 + self.config.clip_epsilon), 1 - self.config.clip_epsilon) * adv

            policy_loss -= min(surr1, surr2)

            # Value loss
            value_pred = self.value_network.forward(state_vector)
            value_loss += (value_pred - ret) ** 2

            # Entropy bonus (encourage exploration)
            for p in probs:
                if p > 1e-8:
                    entropy -= p * math.log(p)

        n = len(batch)
        policy_loss /= n
        value_loss /= n
        entropy /= n

        # Update networks using evolution strategies (simplified)
        self._es_update_policy(batch, advantages)
        self._es_update_value(batch, returns)

        return policy_loss, value_loss, entropy

    def _es_update_policy(
        self,
        batch: list[Experience],
        advantages: list[float],
    ) -> None:
        """Update policy network using evolution strategies."""
        params = self.policy_network.get_parameters()
        num_params = len(params)

        # Generate perturbations
        sigma = 0.1
        num_samples = 10

        best_params = params.copy()
        best_score = float("-inf")

        for _ in range(num_samples):
            # Random perturbation
            noise = [random.gauss(0, sigma) for _ in range(num_params)]
            perturbed = [p + n for p, n in zip(params, noise)]

            # Evaluate perturbed policy
            self.policy_network.set_parameters(perturbed)
            score = 0.0

            for exp, adv in zip(batch, advantages):
                state_vector = exp.state.to_vector()
                probs = self.policy_network.forward(state_vector)
                log_prob = sum(w * math.log(max(p, 1e-8)) for w, p in zip(exp.action, probs))
                score += log_prob * adv

            if score > best_score:
                best_score = score
                best_params = perturbed.copy()

        # Update with best parameters (with learning rate)
        updated = [p + self.config.learning_rate * (bp - p) for p, bp in zip(params, best_params)]
        self.policy_network.set_parameters(updated)

    def _es_update_value(
        self,
        batch: list[Experience],
        returns: list[float],
    ) -> None:
        """Update value network using evolution strategies."""
        params = self.value_network.get_parameters()
        num_params = len(params)

        sigma = 0.1
        num_samples = 10

        best_params = params.copy()
        best_loss = float("inf")

        for _ in range(num_samples):
            noise = [random.gauss(0, sigma) for _ in range(num_params)]
            perturbed = [p + n for p, n in zip(params, noise)]

            self.value_network.set_parameters(perturbed)
            loss = 0.0

            for exp, ret in zip(batch, returns):
                state_vector = exp.state.to_vector()
                value_pred = self.value_network.forward(state_vector)
                loss += (value_pred - ret) ** 2

            if loss < best_loss:
                best_loss = loss
                best_params = perturbed.copy()

        updated = [p + self.config.learning_rate * (bp - p) for p, bp in zip(params, best_params)]
        self.value_network.set_parameters(updated)

    def get_statistics(self) -> dict[str, Any]:
        """Get optimizer statistics."""
        avg_reward = 0.0
        if self.total_rewards:
            avg_reward = sum(self.total_rewards[-100:]) / len(self.total_rewards[-100:])

        current_weights = []
        if self.avg_weights_history:
            current_weights = self.avg_weights_history[-1]

        return {
            "num_experiences": len(self.buffer),
            "training_steps": self.training_step,
            "total_outcomes": len(self.total_rewards),
            "avg_reward_100": avg_reward,
            "current_avg_weights": current_weights,
            "model_names": self.model_names,
            "config": {
                "learning_rate": self.config.learning_rate,
                "clip_epsilon": self.config.clip_epsilon,
                "gamma": self.config.gamma,
            },
        }

    def save_state(self) -> dict[str, Any]:
        """Save optimizer state for persistence."""
        return {
            "policy_params": self.policy_network.get_parameters(),
            "value_params": self.value_network.get_parameters(),
            "training_step": self.training_step,
            "total_rewards": self.total_rewards[-1000:],  # Keep last 1000
            "avg_weights_history": self.avg_weights_history[-100:],
            "returns_history": self.reward_calculator.returns_history,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load optimizer state."""
        if "policy_params" in state:
            self.policy_network.set_parameters(state["policy_params"])
        if "value_params" in state:
            self.value_network.set_parameters(state["value_params"])
        if "training_step" in state:
            self.training_step = state["training_step"]
        if "total_rewards" in state:
            self.total_rewards = state["total_rewards"]
        if "avg_weights_history" in state:
            self.avg_weights_history = state["avg_weights_history"]
        if "returns_history" in state:
            self.reward_calculator.returns_history = state["returns_history"]

    def reset(self) -> None:
        """Reset optimizer state."""
        self.buffer.clear()
        self.reward_calculator.reset()
        self.total_rewards = []
        self.avg_weights_history = []
        self.training_step = 0


def create_ppo_optimizer(
    num_models: int = 3,
    model_names: list[str] | None = None,
    learning_rate: float = 0.001,
    reward_type: RewardType = RewardType.COMBINED,
    buffer_size: int = 1000,
) -> PPOWeightOptimizer:
    """
    Factory function to create PPO weight optimizer.

    Args:
        num_models: Number of models in ensemble
        model_names: Names of models
        learning_rate: Learning rate
        reward_type: Type of reward to use
        buffer_size: Experience buffer size

    Returns:
        Configured PPOWeightOptimizer
    """
    config = PPOConfig(
        learning_rate=learning_rate,
        reward_type=reward_type,
        buffer_size=buffer_size,
    )
    return PPOWeightOptimizer(
        num_models=num_models,
        model_names=model_names,
        config=config,
    )


def create_adaptive_weight_optimizer(
    model_names: list[str],
    sharpe_weight: float = 0.4,
    returns_weight: float = 0.3,
    accuracy_weight: float = 0.3,
    min_weight: float = 0.05,
    max_weight: float = 0.90,
) -> PPOWeightOptimizer:
    """
    Factory function for trading-focused weight optimizer.

    Args:
        model_names: Names of ensemble models
        sharpe_weight: Weight for Sharpe-based reward
        returns_weight: Weight for returns-based reward
        accuracy_weight: Weight for accuracy-based reward
        min_weight: Minimum weight per model
        max_weight: Maximum weight per model

    Returns:
        Configured PPOWeightOptimizer for trading
    """
    config = PPOConfig(
        reward_type=RewardType.COMBINED,
        sharpe_weight=sharpe_weight,
        returns_weight=returns_weight,
        accuracy_weight=accuracy_weight,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    return PPOWeightOptimizer(
        num_models=len(model_names),
        model_names=model_names,
        config=config,
    )
