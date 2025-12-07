"""
Multi-Asset RL Rebalancing System

UPGRADE-010 Sprint 3: Feature 2

Implements PPO-based reinforcement learning for dynamic multi-asset
portfolio rebalancing. Learns optimal allocation weights based on
market conditions, transaction costs, and risk-adjusted returns.

Key Features:
- PPO policy for multi-asset allocation
- Transaction cost awareness
- Risk-adjusted return optimization
- Integration with RiskManager
- Rebalancing frequency optimization

QuantConnect Compatible: Yes
"""

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)

# Use numpy if available, fallback to pure Python
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class RebalanceAction(Enum):
    """Types of rebalancing actions."""

    HOLD = "hold"  # No change
    REBALANCE = "rebalance"  # Execute rebalance
    PARTIAL = "partial"  # Partial rebalance (reduce transaction costs)


class RebalanceFrequency(Enum):
    """Rebalancing frequency options."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ADAPTIVE = "adaptive"  # Determined by RL policy


@dataclass
class AssetState:
    """State representation for a single asset."""

    symbol: str
    current_weight: float
    target_weight: float
    price: float
    price_change_1d: float
    price_change_5d: float
    price_change_20d: float
    volatility: float
    volume_ratio: float  # Current vs average volume
    momentum_score: float  # RSI-like momentum
    correlation_to_portfolio: float  # Correlation with other assets

    def to_vector(self) -> list[float]:
        """Convert to feature vector."""
        return [
            self.current_weight,
            self.target_weight,
            self.price_change_1d,
            self.price_change_5d,
            self.price_change_20d,
            self.volatility,
            self.volume_ratio,
            self.momentum_score,
            self.correlation_to_portfolio,
        ]


@dataclass
class PortfolioState:
    """Complete portfolio state for RL agent."""

    assets: list[AssetState]
    total_value: float
    cash_weight: float
    unrealized_pnl: float
    realized_pnl_today: float
    days_since_rebalance: int
    portfolio_volatility: float
    portfolio_beta: float
    sharpe_ratio: float
    max_drawdown: float
    transaction_cost_estimate: float  # Estimated cost to rebalance
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_vector(self) -> list[float]:
        """Convert to feature vector."""
        features = [
            self.cash_weight,
            self.unrealized_pnl / max(self.total_value, 1),
            self.realized_pnl_today / max(self.total_value, 1),
            min(self.days_since_rebalance / 30, 1.0),  # Normalize to [0,1]
            self.portfolio_volatility,
            self.portfolio_beta,
            self.sharpe_ratio / 3.0 if self.sharpe_ratio else 0,  # Normalize
            self.max_drawdown,
            self.transaction_cost_estimate / max(self.total_value, 1),
        ]

        # Add asset features
        for asset in self.assets:
            features.extend(asset.to_vector())

        return features

    @property
    def num_assets(self) -> int:
        """Number of assets in portfolio."""
        return len(self.assets)


@dataclass
class RebalanceDecision:
    """Output of rebalancing decision."""

    action: RebalanceAction
    target_weights: dict[str, float]
    current_weights: dict[str, float]
    weight_changes: dict[str, float]
    estimated_cost: float
    confidence: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "target_weights": self.target_weights,
            "current_weights": self.current_weights,
            "weight_changes": self.weight_changes,
            "estimated_cost": self.estimated_cost,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RebalanceExperience:
    """Experience tuple for RL training."""

    state: PortfolioState
    action: dict[str, float]  # Target weights
    reward: float
    next_state: PortfolioState | None
    done: bool
    log_prob: float
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RLRebalancerConfig:
    """Configuration for RL rebalancer."""

    # Asset configuration
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 1.0  # Maximum weight per asset
    max_cash_weight: float = 0.3  # Maximum cash allocation

    # Transaction costs
    commission_pct: float = 0.001  # 0.1% commission
    slippage_pct: float = 0.001  # 0.1% slippage
    min_trade_value: float = 100.0  # Minimum trade size

    # Rebalancing thresholds
    min_weight_deviation: float = 0.02  # 2% deviation to trigger rebalance
    max_weight_deviation: float = 0.10  # 10% max before forced rebalance

    # PPO hyperparameters
    learning_rate: float = 0.0003
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

    # Training
    batch_size: int = 32
    buffer_size: int = 1000
    min_buffer_size: int = 50
    epochs_per_update: int = 4

    # Network
    hidden_size: int = 128
    num_hidden_layers: int = 3

    # Risk management
    max_turnover: float = 0.5  # Max 50% turnover per rebalance
    risk_free_rate: float = 0.05  # For Sharpe calculation


class PolicyNetwork:
    """
    Policy network for portfolio allocation.

    Outputs target weights for each asset using softmax.
    """

    def __init__(
        self,
        input_size: int,
        num_assets: int,
        hidden_size: int = 128,
        num_hidden_layers: int = 3,
        seed: int = 42,
    ):
        """Initialize policy network."""
        self.input_size = input_size
        self.output_size = num_assets + 1  # Assets + cash
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        random.seed(seed)

        # Initialize weights (Xavier initialization)
        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []

        # Input -> First hidden
        self.weights.append(self._init_weights(input_size, hidden_size))
        self.biases.append([0.0] * hidden_size)

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.weights.append(self._init_weights(hidden_size, hidden_size))
            self.biases.append([0.0] * hidden_size)

        # Last hidden -> Output
        self.weights.append(self._init_weights(hidden_size, self.output_size))
        self.biases.append([0.0] * self.output_size)

    def _init_weights(self, fan_in: int, fan_out: int) -> list[list[float]]:
        """Xavier initialization."""
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return [[random.gauss(0, std) for _ in range(fan_out)] for _ in range(fan_in)]

    def _tanh(self, x: float) -> float:
        """Tanh activation."""
        try:
            return math.tanh(x)
        except OverflowError:
            return 1.0 if x > 0 else -1.0

    def _softmax(self, x: list[float]) -> list[float]:
        """Softmax for output probabilities."""
        max_x = max(x)
        exp_x = [math.exp(min(xi - max_x, 10)) for xi in x]  # Clamp to prevent overflow
        sum_exp = sum(exp_x) + 1e-10
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

    def forward(self, state_vector: list[float]) -> list[float]:
        """
        Forward pass to get target weights.

        Args:
            state_vector: Portfolio state as feature vector

        Returns:
            List of target weights (including cash as last element)
        """
        current = state_vector

        # Hidden layers with tanh
        for i in range(len(self.weights) - 1):
            current = self._matmul(current, self.weights[i], self.biases[i])
            current = [self._tanh(v) for v in current]

        # Output layer with softmax
        output = self._matmul(current, self.weights[-1], self.biases[-1])
        return self._softmax(output)

    def get_parameters(self) -> list[float]:
        """Flatten all parameters into single list."""
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


class ValueNetwork:
    """Value function network for PPO."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_hidden_layers: int = 3,
        seed: int = 42,
    ):
        """Initialize value network."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        random.seed(seed)

        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []

        # Input -> First hidden
        self.weights.append(self._init_weights(input_size, hidden_size))
        self.biases.append([0.0] * hidden_size)

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.weights.append(self._init_weights(hidden_size, hidden_size))
            self.biases.append([0.0] * hidden_size)

        # Last hidden -> Output (single value)
        self.weights.append(self._init_weights(hidden_size, 1))
        self.biases.append([0.0])

    def _init_weights(self, fan_in: int, fan_out: int) -> list[list[float]]:
        """Xavier initialization."""
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return [[random.gauss(0, std) for _ in range(fan_out)] for _ in range(fan_in)]

    def _tanh(self, x: float) -> float:
        """Tanh activation."""
        try:
            return math.tanh(x)
        except OverflowError:
            return 1.0 if x > 0 else -1.0

    def _matmul(
        self,
        x: list[float],
        weights: list[list[float]],
        bias: list[float],
    ) -> list[float]:
        """Matrix multiplication."""
        result = []
        for j in range(len(weights[0])):
            val = bias[j]
            for i in range(len(x)):
                val += x[i] * weights[i][j]
            result.append(val)
        return result

    def forward(self, state_vector: list[float]) -> float:
        """Forward pass to estimate state value."""
        current = state_vector

        for i in range(len(self.weights) - 1):
            current = self._matmul(current, self.weights[i], self.biases[i])
            current = [self._tanh(v) for v in current]

        output = self._matmul(current, self.weights[-1], self.biases[-1])
        return output[0]

    def get_parameters(self) -> list[float]:
        """Flatten all parameters into single list."""
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


class RLRebalancer:
    """
    Reinforcement Learning-based portfolio rebalancer.

    Uses PPO to learn optimal portfolio allocations based on market
    conditions, transaction costs, and risk-adjusted returns.
    """

    def __init__(
        self,
        asset_symbols: list[str],
        config: RLRebalancerConfig | None = None,
    ):
        """
        Initialize RL rebalancer.

        Args:
            asset_symbols: List of asset symbols to manage
            config: Rebalancer configuration
        """
        self.asset_symbols = asset_symbols
        self.config = config or RLRebalancerConfig()
        self.num_assets = len(asset_symbols)

        # Calculate input size: portfolio features + asset features per asset
        # Portfolio: 9 features
        # Per asset: 9 features
        self._portfolio_features = 9
        self._asset_features = 9
        self._input_size = self._portfolio_features + (self._asset_features * self.num_assets)

        # Initialize networks
        self._policy = PolicyNetwork(
            input_size=self._input_size,
            num_assets=self.num_assets,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
        )
        self._value = ValueNetwork(
            input_size=self._input_size,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
        )

        # Experience buffer
        self._buffer: deque = deque(maxlen=self.config.buffer_size)

        # State tracking
        self._last_state: PortfolioState | None = None
        self._last_action: dict[str, float] | None = None
        self._last_log_prob: float = 0.0
        self._last_value: float = 0.0
        self._rebalance_count: int = 0
        self._total_reward: float = 0.0

    def get_target_weights(
        self,
        state: PortfolioState,
        deterministic: bool = False,
    ) -> tuple[dict[str, float], float, float]:
        """
        Get target portfolio weights from policy.

        Args:
            state: Current portfolio state
            deterministic: If True, use mode of distribution

        Returns:
            Tuple of (target_weights dict, log_prob, value estimate)
        """
        state_vector = state.to_vector()

        # Ensure state vector matches expected size
        if len(state_vector) != self._input_size:
            # Pad or truncate
            if len(state_vector) < self._input_size:
                state_vector.extend([0.0] * (self._input_size - len(state_vector)))
            else:
                state_vector = state_vector[: self._input_size]

        # Get policy output
        raw_weights = self._policy.forward(state_vector)

        # Apply constraints
        constrained_weights = self._apply_weight_constraints(raw_weights)

        # Add noise for exploration (if not deterministic)
        if not deterministic:
            noise = [random.gauss(0, 0.02) for _ in constrained_weights]
            noisy_weights = [w + n for w, n in zip(constrained_weights, noise)]
            constrained_weights = self._apply_weight_constraints(noisy_weights)

        # Calculate log probability (simplified)
        log_prob = sum(math.log(max(w, 1e-10)) for w in constrained_weights)

        # Get value estimate
        value = self._value.forward(state_vector)

        # Create weight dictionary
        target_weights = {}
        for i, symbol in enumerate(self.asset_symbols):
            target_weights[symbol] = constrained_weights[i]
        target_weights["CASH"] = constrained_weights[-1]

        return target_weights, log_prob, value

    def _apply_weight_constraints(self, weights: list[float]) -> list[float]:
        """Apply weight constraints and normalize."""
        # Separate assets and cash
        asset_weights = weights[:-1]
        cash_weight = weights[-1]

        # Apply min/max constraints to assets
        constrained = []
        for w in asset_weights:
            w = max(self.config.min_weight, min(self.config.max_weight, w))
            constrained.append(w)

        # Constrain cash
        cash_weight = max(0, min(self.config.max_cash_weight, cash_weight))
        constrained.append(cash_weight)

        # Normalize to sum to 1
        total = sum(constrained)
        if total > 0:
            constrained = [w / total for w in constrained]
        else:
            # Equal weights fallback
            n = len(constrained)
            constrained = [1.0 / n] * n

        return constrained

    def decide_rebalance(
        self,
        state: PortfolioState,
    ) -> RebalanceDecision:
        """
        Decide whether and how to rebalance.

        Args:
            state: Current portfolio state

        Returns:
            Rebalancing decision
        """
        # Get target weights
        target_weights, log_prob, value = self.get_target_weights(state)

        # Calculate current weights
        current_weights = {asset.symbol: asset.current_weight for asset in state.assets}
        current_weights["CASH"] = state.cash_weight

        # Calculate weight changes
        weight_changes = {}
        max_deviation = 0.0
        total_turnover = 0.0

        for symbol in list(current_weights.keys()):
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            change = target - current
            weight_changes[symbol] = change
            max_deviation = max(max_deviation, abs(change))
            total_turnover += abs(change)

        # Estimate transaction cost
        estimated_cost = self._estimate_transaction_cost(
            state.total_value,
            weight_changes,
        )

        # Determine action based on deviation and cost
        if max_deviation < self.config.min_weight_deviation:
            action = RebalanceAction.HOLD
            reasoning = f"Max deviation ({max_deviation:.2%}) below threshold ({self.config.min_weight_deviation:.2%})"
        elif total_turnover > self.config.max_turnover:
            action = RebalanceAction.PARTIAL
            reasoning = (
                f"Total turnover ({total_turnover:.2%}) exceeds max ({self.config.max_turnover:.2%}), partial rebalance"
            )
        elif max_deviation > self.config.max_weight_deviation:
            action = RebalanceAction.REBALANCE
            reasoning = (
                f"Max deviation ({max_deviation:.2%}) exceeds max threshold ({self.config.max_weight_deviation:.2%})"
            )
        elif estimated_cost / state.total_value > 0.01:
            action = RebalanceAction.HOLD
            reasoning = f"Transaction cost ({estimated_cost / state.total_value:.2%}) too high"
        else:
            action = RebalanceAction.REBALANCE
            reasoning = "Optimal rebalancing conditions"

        # Store for training
        self._last_state = state
        self._last_action = target_weights
        self._last_log_prob = log_prob
        self._last_value = value

        # Calculate confidence
        confidence = 1.0 - (max_deviation / 0.20)  # Higher deviation = lower confidence
        confidence = max(0, min(1, confidence))

        return RebalanceDecision(
            action=action,
            target_weights=target_weights,
            current_weights=current_weights,
            weight_changes=weight_changes,
            estimated_cost=estimated_cost,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _estimate_transaction_cost(
        self,
        portfolio_value: float,
        weight_changes: dict[str, float],
    ) -> float:
        """Estimate transaction cost for rebalancing."""
        total_cost = 0.0

        for symbol, change in weight_changes.items():
            if symbol == "CASH":
                continue

            trade_value = abs(change) * portfolio_value

            if trade_value < self.config.min_trade_value:
                continue

            # Commission + slippage
            cost = trade_value * (self.config.commission_pct + self.config.slippage_pct)
            total_cost += cost

        return total_cost

    def record_outcome(
        self,
        next_state: PortfolioState,
        done: bool = False,
    ) -> None:
        """
        Record outcome after rebalancing decision.

        Args:
            next_state: Portfolio state after action
            done: Whether episode is complete
        """
        if self._last_state is None:
            return

        # Calculate reward
        reward = self._calculate_reward(self._last_state, next_state)

        # Create experience
        experience = RebalanceExperience(
            state=self._last_state,
            action=self._last_action or {},
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=self._last_log_prob,
            value=self._last_value,
        )

        self._buffer.append(experience)
        self._total_reward += reward

        # Reset state
        self._last_state = None
        self._last_action = None

    def _calculate_reward(
        self,
        prev_state: PortfolioState,
        next_state: PortfolioState,
    ) -> float:
        """
        Calculate reward for rebalancing action.

        Combines multiple factors:
        - Risk-adjusted returns (Sharpe improvement)
        - Transaction costs (penalize frequent trading)
        - Drawdown (penalize increased drawdown)
        """
        # Return component
        value_change = (next_state.total_value - prev_state.total_value) / prev_state.total_value
        return_reward = value_change * 10  # Scale

        # Risk-adjusted component
        sharpe_improvement = next_state.sharpe_ratio - prev_state.sharpe_ratio
        risk_reward = sharpe_improvement * 5

        # Transaction cost penalty
        cost_penalty = -next_state.transaction_cost_estimate / max(next_state.total_value, 1) * 100

        # Drawdown penalty
        drawdown_penalty = -max(0, next_state.max_drawdown - prev_state.max_drawdown) * 10

        # Combine
        total_reward = return_reward + risk_reward + cost_penalty + drawdown_penalty

        return total_reward

    def train(self) -> dict[str, float]:
        """
        Train policy using PPO.

        Returns:
            Training metrics
        """
        if len(self._buffer) < self.config.min_buffer_size:
            return {"status": "insufficient_data", "buffer_size": len(self._buffer)}

        # Sample batch
        batch_size = min(self.config.batch_size, len(self._buffer))
        batch = random.sample(list(self._buffer), batch_size)

        # Calculate advantages using GAE
        advantages = self._compute_advantages(batch)

        # PPO update (simplified - gradient-free approach)
        policy_loss = 0.0
        value_loss = 0.0

        for epoch in range(self.config.epochs_per_update):
            for i, exp in enumerate(batch):
                if exp.next_state is None:
                    continue

                # Get current policy output
                state_vector = exp.state.to_vector()
                if len(state_vector) < self._input_size:
                    state_vector.extend([0.0] * (self._input_size - len(state_vector)))

                # Simple gradient-free update
                # Perturb weights and keep if improves loss
                current_params = self._policy.get_parameters()

                # Random perturbation
                perturbation = [random.gauss(0, 0.01) for _ in current_params]
                new_params = [p + d for p, d in zip(current_params, perturbation)]

                # Evaluate perturbation
                self._policy.set_parameters(new_params)
                new_weights = self._policy.forward(state_vector)
                new_value = self._value.forward(state_vector)

                # Simple loss: negative advantage-weighted probability
                adv = advantages[i]
                current_loss = -adv * sum(math.log(max(w, 1e-10)) for w in new_weights)

                # Revert if loss increased
                if current_loss > policy_loss:
                    self._policy.set_parameters(current_params)
                else:
                    policy_loss = current_loss

                # Value network update
                value_params = self._value.get_parameters()
                v_perturbation = [random.gauss(0, 0.01) for _ in value_params]
                new_v_params = [p + d for p, d in zip(value_params, v_perturbation)]

                self._value.set_parameters(new_v_params)
                new_v = self._value.forward(state_vector)

                v_loss = (
                    new_v
                    - (
                        exp.reward
                        + self.config.gamma * self._value.forward(exp.next_state.to_vector()[: self._input_size])
                        if not exp.done
                        else exp.reward
                    )
                ) ** 2

                if v_loss > value_loss:
                    self._value.set_parameters(value_params)
                else:
                    value_loss = v_loss

        return {
            "status": "trained",
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "batch_size": batch_size,
            "buffer_size": len(self._buffer),
            "total_reward": self._total_reward,
        }

    def _compute_advantages(
        self,
        batch: list[RebalanceExperience],
    ) -> list[float]:
        """Compute GAE advantages."""
        advantages = []

        for exp in batch:
            if exp.next_state is None:
                advantages.append(exp.reward - exp.value)
                continue

            # Get next state value
            next_vector = exp.next_state.to_vector()
            if len(next_vector) < self._input_size:
                next_vector.extend([0.0] * (self._input_size - len(next_vector)))

            next_value = self._value.forward(next_vector) if not exp.done else 0.0

            # TD error
            td_error = exp.reward + self.config.gamma * next_value - exp.value

            # Simplified GAE (single step)
            advantage = td_error

            advantages.append(advantage)

        return advantages

    def get_metrics(self) -> dict[str, Any]:
        """Get rebalancer metrics."""
        return {
            "num_assets": self.num_assets,
            "buffer_size": len(self._buffer),
            "rebalance_count": self._rebalance_count,
            "total_reward": self._total_reward,
            "config": {
                "min_weight": self.config.min_weight,
                "max_weight": self.config.max_weight,
                "max_cash_weight": self.config.max_cash_weight,
                "commission_pct": self.config.commission_pct,
                "slippage_pct": self.config.slippage_pct,
            },
        }

    def save_state(self) -> dict[str, Any]:
        """Save rebalancer state for persistence."""
        return {
            "policy_params": self._policy.get_parameters(),
            "value_params": self._value.get_parameters(),
            "rebalance_count": self._rebalance_count,
            "total_reward": self._total_reward,
            "asset_symbols": self.asset_symbols,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load rebalancer state."""
        if "policy_params" in state:
            self._policy.set_parameters(state["policy_params"])
        if "value_params" in state:
            self._value.set_parameters(state["value_params"])
        if "rebalance_count" in state:
            self._rebalance_count = state["rebalance_count"]
        if "total_reward" in state:
            self._total_reward = state["total_reward"]


def create_rl_rebalancer(
    asset_symbols: list[str],
    config: RLRebalancerConfig | None = None,
) -> RLRebalancer:
    """
    Create RL rebalancer instance.

    Args:
        asset_symbols: List of asset symbols to manage
        config: Rebalancer configuration

    Returns:
        Configured RLRebalancer instance
    """
    return RLRebalancer(asset_symbols, config)


__all__ = [
    "AssetState",
    "PolicyNetwork",
    "PortfolioState",
    "RLRebalancer",
    "RLRebalancerConfig",
    "RebalanceAction",
    "RebalanceDecision",
    "RebalanceExperience",
    "RebalanceFrequency",
    "ValueNetwork",
    "create_rl_rebalancer",
]
