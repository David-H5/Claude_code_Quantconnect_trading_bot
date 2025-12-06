"""
Multi-Agent Consensus Mechanism

UPGRADE-014 Feature 6: Multi-Agent Architecture

Aggregates opinions from multiple specialized agents into a weighted consensus
for trading decisions. Based on TradingAgents (2024) multi-agent architecture.

Key Features:
- Weighted aggregation of agent opinions
- Confidence-adjusted voting
- Agreement score calculation
- Disagreement detection
- Market regime-aware weighting

QuantConnect Compatible: Yes
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ConsensusSignal(Enum):
    """Consensus trading signal."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    HOLD = "hold"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    CONFLICTED = "conflicted"  # Significant disagreement


class AgentType(Enum):
    """Types of agents that can participate in consensus."""

    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    NEWS = "news"
    RISK = "risk"
    FUNDAMENTAL = "fundamental"
    MOMENTUM = "momentum"
    CUSTOM = "custom"


@dataclass
class AgentOpinion:
    """Single agent's opinion for consensus."""

    agent_type: AgentType
    agent_name: str
    signal_score: float  # -1.0 (bearish) to +1.0 (bullish)
    confidence: float  # 0.0 to 1.0
    reasoning: str
    key_factors: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_type": self.agent_type.value,
            "agent_name": self.agent_name,
            "signal_score": self.signal_score,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "risk_factors": self.risk_factors,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentOpinion":
        """Create from dictionary."""
        return cls(
            agent_type=AgentType(data.get("agent_type", "custom")),
            agent_name=data.get("agent_name", "unknown"),
            signal_score=data.get("signal_score", 0.0),
            confidence=data.get("confidence", 0.5),
            reasoning=data.get("reasoning", ""),
            key_factors=data.get("key_factors", []),
            risk_factors=data.get("risk_factors", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConsensusResult:
    """Result of multi-agent consensus calculation."""

    symbol: str
    signal: ConsensusSignal
    consensus_score: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    agreement_score: float  # 0.0 to 1.0 (how much agents agree)
    participating_agents: int
    agent_opinions: list[AgentOpinion]
    bullish_agents: list[str]
    bearish_agents: list[str]
    neutral_agents: list[str]
    key_bull_factors: list[str]
    key_bear_factors: list[str]
    risk_factors: list[str]
    recommendation: str
    requires_debate: bool  # True if significant disagreement
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "signal": self.signal.value,
            "consensus_score": self.consensus_score,
            "confidence": self.confidence,
            "agreement_score": self.agreement_score,
            "participating_agents": self.participating_agents,
            "agent_opinions": [o.to_dict() for o in self.agent_opinions],
            "bullish_agents": self.bullish_agents,
            "bearish_agents": self.bearish_agents,
            "neutral_agents": self.neutral_agents,
            "key_bull_factors": self.key_bull_factors,
            "key_bear_factors": self.key_bear_factors,
            "risk_factors": self.risk_factors,
            "recommendation": self.recommendation,
            "requires_debate": self.requires_debate,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConsensusConfig:
    """Configuration for consensus calculation."""

    # Agent weights (must sum to 1.0 for normalized calculation)
    default_weights: dict[AgentType, float] = field(
        default_factory=lambda: {
            AgentType.SENTIMENT: 0.25,
            AgentType.TECHNICAL: 0.30,
            AgentType.NEWS: 0.20,
            AgentType.RISK: 0.15,
            AgentType.FUNDAMENTAL: 0.10,
        }
    )

    # Signal thresholds
    strong_signal_threshold: float = 0.7  # Above this = strong buy/sell
    weak_signal_threshold: float = 0.3  # Below this = weak signal
    neutral_threshold: float = 0.15  # Within +/- this = hold

    # Agreement thresholds
    high_agreement_threshold: float = 0.8  # Agents mostly agree
    debate_trigger_threshold: float = 0.4  # Low agreement triggers debate

    # Minimum requirements
    min_agents_for_consensus: int = 2
    min_confidence_threshold: float = 0.3

    # Confidence weighting
    use_confidence_weighting: bool = True
    confidence_weight_power: float = 1.5  # Higher = more weight to confident agents

    # Market regime adjustments
    high_vol_sentiment_weight_mult: float = 0.7  # Reduce sentiment in high vol
    high_vol_technical_weight_mult: float = 1.2  # Increase technical in high vol


class MultiAgentConsensus:
    """
    Multi-agent consensus mechanism for trading decisions.

    Aggregates opinions from multiple specialized agents (sentiment, technical,
    news, risk, etc.) into a weighted consensus signal.

    Based on TradingAgents (2024) research showing improved decision quality
    through multi-agent collaboration.

    Usage:
        consensus = MultiAgentConsensus()

        # Add agent opinions
        consensus.add_opinion(AgentOpinion(
            agent_type=AgentType.SENTIMENT,
            agent_name="SentimentAnalyst",
            signal_score=0.6,
            confidence=0.8,
            reasoning="Bullish news sentiment"
        ))

        consensus.add_opinion(AgentOpinion(
            agent_type=AgentType.TECHNICAL,
            agent_name="TechnicalAnalyst",
            signal_score=0.3,
            confidence=0.7,
            reasoning="Price above key support"
        ))

        # Calculate consensus
        result = consensus.calculate_consensus("AAPL")

        print(f"Signal: {result.signal.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Agreement: {result.agreement_score:.2f}")
    """

    def __init__(
        self,
        config: ConsensusConfig | None = None,
        custom_weights: dict[AgentType, float] | None = None,
    ):
        """
        Initialize multi-agent consensus.

        Args:
            config: Configuration for consensus calculation
            custom_weights: Optional custom agent weights
        """
        self.config = config or ConsensusConfig()
        self.opinions: list[AgentOpinion] = []

        # Override default weights if custom provided
        if custom_weights:
            self.weights = custom_weights
        else:
            self.weights = self.config.default_weights.copy()

        # Track history
        self.consensus_history: list[ConsensusResult] = []
        self._consensus_count = 0

    def add_opinion(self, opinion: AgentOpinion) -> None:
        """
        Add an agent opinion for consensus.

        Args:
            opinion: Agent's opinion
        """
        self.opinions.append(opinion)

    def add_opinions(self, opinions: list[AgentOpinion]) -> None:
        """
        Add multiple agent opinions.

        Args:
            opinions: List of agent opinions
        """
        self.opinions.extend(opinions)

    def clear_opinions(self) -> None:
        """Clear all current opinions."""
        self.opinions = []

    def set_weights(self, weights: dict[AgentType, float]) -> None:
        """
        Set custom agent weights.

        Args:
            weights: Dictionary mapping agent types to weights
        """
        self.weights = weights

    def adjust_weights_for_regime(
        self,
        is_high_volatility: bool = False,
        is_trending: bool = False,
    ) -> None:
        """
        Adjust weights based on market regime.

        Args:
            is_high_volatility: True if market is in high volatility regime
            is_trending: True if market is trending
        """
        if is_high_volatility:
            # In high vol, reduce sentiment weight, increase technical
            if AgentType.SENTIMENT in self.weights:
                self.weights[AgentType.SENTIMENT] *= self.config.high_vol_sentiment_weight_mult
            if AgentType.TECHNICAL in self.weights:
                self.weights[AgentType.TECHNICAL] *= self.config.high_vol_technical_weight_mult
            if AgentType.RISK in self.weights:
                self.weights[AgentType.RISK] *= 1.3  # More risk focus in high vol

            # Normalize
            self._normalize_weights()

        if is_trending:
            # In trending markets, increase technical/momentum weight
            if AgentType.TECHNICAL in self.weights:
                self.weights[AgentType.TECHNICAL] *= 1.1
            if AgentType.MOMENTUM in self.weights:
                self.weights[AgentType.MOMENTUM] *= 1.2

            self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total

    def calculate_consensus(
        self,
        symbol: str,
        is_high_volatility: bool = False,
    ) -> ConsensusResult:
        """
        Calculate consensus from all agent opinions.

        Args:
            symbol: Symbol being analyzed
            is_high_volatility: Whether market is in high volatility

        Returns:
            ConsensusResult with aggregated signal
        """
        if len(self.opinions) < self.config.min_agents_for_consensus:
            return self._create_insufficient_data_result(symbol)

        # Filter opinions by minimum confidence
        valid_opinions = [o for o in self.opinions if o.confidence >= self.config.min_confidence_threshold]

        if len(valid_opinions) < self.config.min_agents_for_consensus:
            return self._create_insufficient_data_result(symbol)

        # Calculate weighted consensus score
        consensus_score, total_weight = self._calculate_weighted_score(valid_opinions)

        # Calculate agreement score
        agreement_score = self._calculate_agreement(valid_opinions)

        # Calculate overall confidence
        confidence = self._calculate_confidence(valid_opinions, agreement_score)

        # Determine signal
        signal = self._determine_signal(consensus_score, agreement_score)

        # Categorize agents
        bullish_agents = [o.agent_name for o in valid_opinions if o.signal_score > 0.2]
        bearish_agents = [o.agent_name for o in valid_opinions if o.signal_score < -0.2]
        neutral_agents = [o.agent_name for o in valid_opinions if -0.2 <= o.signal_score <= 0.2]

        # Aggregate factors
        key_bull_factors = self._aggregate_factors(valid_opinions, bullish=True)
        key_bear_factors = self._aggregate_factors(valid_opinions, bullish=False)
        risk_factors = self._aggregate_risks(valid_opinions)

        # Generate recommendation
        recommendation = self._generate_recommendation(signal, consensus_score, confidence, agreement_score)

        # Check if debate is needed
        requires_debate = agreement_score < self.config.debate_trigger_threshold

        result = ConsensusResult(
            symbol=symbol,
            signal=signal,
            consensus_score=consensus_score,
            confidence=confidence,
            agreement_score=agreement_score,
            participating_agents=len(valid_opinions),
            agent_opinions=valid_opinions,
            bullish_agents=bullish_agents,
            bearish_agents=bearish_agents,
            neutral_agents=neutral_agents,
            key_bull_factors=key_bull_factors,
            key_bear_factors=key_bear_factors,
            risk_factors=risk_factors,
            recommendation=recommendation,
            requires_debate=requires_debate,
        )

        # Record history
        self._consensus_count += 1
        self.consensus_history.append(result)

        return result

    def _calculate_weighted_score(
        self,
        opinions: list[AgentOpinion],
    ) -> tuple[float, float]:
        """Calculate weighted consensus score."""
        total_weight = 0.0
        weighted_sum = 0.0

        for opinion in opinions:
            # Get base weight for agent type
            base_weight = self.weights.get(opinion.agent_type, 0.1)

            # Apply confidence weighting if enabled
            if self.config.use_confidence_weighting:
                confidence_mult = opinion.confidence**self.config.confidence_weight_power
                weight = base_weight * confidence_mult
            else:
                weight = base_weight

            weighted_sum += opinion.signal_score * weight
            total_weight += weight

        if total_weight > 0:
            consensus_score = weighted_sum / total_weight
        else:
            consensus_score = 0.0

        # Clamp to [-1, 1]
        consensus_score = max(-1.0, min(1.0, consensus_score))

        return consensus_score, total_weight

    def _calculate_agreement(self, opinions: list[AgentOpinion]) -> float:
        """
        Calculate agreement score between agents.

        Returns 1.0 if all agents agree perfectly, 0.0 if maximally disagreed.
        """
        if len(opinions) < 2:
            return 1.0

        scores = [o.signal_score for o in opinions]

        # Calculate standard deviation as disagreement measure
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance**0.5

        # Max std_dev is 1.0 (if scores are -1 and +1)
        # Convert to agreement (0 std = 1.0 agreement, 1.0 std = 0.0 agreement)
        agreement = 1.0 - min(std_dev, 1.0)

        # Also consider direction agreement
        bullish_count = sum(1 for s in scores if s > 0.2)
        bearish_count = sum(1 for s in scores if s < -0.2)
        total = len(scores)

        # If all in same direction, boost agreement
        if bullish_count == total or bearish_count == total:
            agreement = min(1.0, agreement * 1.2)
        # If split, reduce agreement
        elif bullish_count > 0 and bearish_count > 0:
            split_ratio = min(bullish_count, bearish_count) / max(bullish_count, bearish_count)
            agreement *= 1.0 - split_ratio * 0.3

        return max(0.0, min(1.0, agreement))

    def _calculate_confidence(
        self,
        opinions: list[AgentOpinion],
        agreement_score: float,
    ) -> float:
        """Calculate overall confidence in consensus."""
        # Base confidence from agent confidences
        avg_confidence = sum(o.confidence for o in opinions) / len(opinions)

        # Weight by agreement
        confidence = avg_confidence * (0.5 + 0.5 * agreement_score)

        # Boost if many agents
        if len(opinions) >= 4:
            confidence *= 1.1
        elif len(opinions) <= 2:
            confidence *= 0.9

        return max(0.0, min(1.0, confidence))

    def _determine_signal(
        self,
        consensus_score: float,
        agreement_score: float,
    ) -> ConsensusSignal:
        """Determine trading signal from consensus score."""
        # If low agreement, mark as conflicted
        if agreement_score < self.config.debate_trigger_threshold:
            return ConsensusSignal.CONFLICTED

        abs_score = abs(consensus_score)

        # Within neutral threshold = hold
        if abs_score <= self.config.neutral_threshold:
            return ConsensusSignal.HOLD

        # Determine direction and strength
        if consensus_score > 0:
            if abs_score >= self.config.strong_signal_threshold:
                return ConsensusSignal.STRONG_BUY
            elif abs_score >= self.config.weak_signal_threshold:
                return ConsensusSignal.BUY
            else:
                return ConsensusSignal.WEAK_BUY
        else:
            if abs_score >= self.config.strong_signal_threshold:
                return ConsensusSignal.STRONG_SELL
            elif abs_score >= self.config.weak_signal_threshold:
                return ConsensusSignal.SELL
            else:
                return ConsensusSignal.WEAK_SELL

    def _aggregate_factors(
        self,
        opinions: list[AgentOpinion],
        bullish: bool = True,
    ) -> list[str]:
        """Aggregate key factors from opinions."""
        factors = []
        for opinion in opinions:
            if (bullish and opinion.signal_score > 0.2) or (not bullish and opinion.signal_score < -0.2):
                factors.extend(opinion.key_factors[:2])

        # Deduplicate and limit
        seen = set()
        unique_factors = []
        for f in factors:
            if f.lower() not in seen:
                seen.add(f.lower())
                unique_factors.append(f)

        return unique_factors[:5]

    def _aggregate_risks(self, opinions: list[AgentOpinion]) -> list[str]:
        """Aggregate risk factors from all opinions."""
        risks = []
        for opinion in opinions:
            risks.extend(opinion.risk_factors[:2])

        # Deduplicate and limit
        seen = set()
        unique_risks = []
        for r in risks:
            if r.lower() not in seen:
                seen.add(r.lower())
                unique_risks.append(r)

        return unique_risks[:5]

    def _generate_recommendation(
        self,
        signal: ConsensusSignal,
        score: float,
        confidence: float,
        agreement: float,
    ) -> str:
        """Generate trading recommendation text."""
        recommendations = {
            ConsensusSignal.STRONG_BUY: "Strong bullish consensus - consider aggressive long entry",
            ConsensusSignal.BUY: "Bullish consensus - consider long entry on pullback",
            ConsensusSignal.WEAK_BUY: "Slightly bullish - small position or wait for confirmation",
            ConsensusSignal.HOLD: "No clear signal - maintain current position",
            ConsensusSignal.WEAK_SELL: "Slightly bearish - reduce exposure or tighten stops",
            ConsensusSignal.SELL: "Bearish consensus - consider reducing/exiting positions",
            ConsensusSignal.STRONG_SELL: "Strong bearish consensus - consider exit or short",
            ConsensusSignal.CONFLICTED: "Agents disagree significantly - wait for clarity or run debate",
        }

        base_rec = recommendations.get(signal, "No recommendation")

        # Add confidence qualifier
        if confidence < 0.5:
            base_rec += " (low confidence)"
        elif confidence > 0.8:
            base_rec += " (high confidence)"

        if agreement < 0.5:
            base_rec += " [WARNING: Low agreement between agents]"

        return base_rec

    def _create_insufficient_data_result(self, symbol: str) -> ConsensusResult:
        """Create result when insufficient data."""
        return ConsensusResult(
            symbol=symbol,
            signal=ConsensusSignal.HOLD,
            consensus_score=0.0,
            confidence=0.0,
            agreement_score=0.0,
            participating_agents=len(self.opinions),
            agent_opinions=self.opinions,
            bullish_agents=[],
            bearish_agents=[],
            neutral_agents=[],
            key_bull_factors=[],
            key_bear_factors=[],
            risk_factors=["Insufficient agent opinions for consensus"],
            recommendation=f"Insufficient data for {symbol} - need at least {self.config.min_agents_for_consensus} agent opinions",
            requires_debate=False,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics on consensus calculations."""
        if not self.consensus_history:
            return {
                "total_calculations": 0,
                "avg_agreement": 0,
                "avg_confidence": 0,
                "signal_distribution": {},
            }

        signals = {}
        total_agreement = 0
        total_confidence = 0

        for result in self.consensus_history:
            sig = result.signal.value
            signals[sig] = signals.get(sig, 0) + 1
            total_agreement += result.agreement_score
            total_confidence += result.confidence

        n = len(self.consensus_history)
        return {
            "total_calculations": n,
            "avg_agreement": total_agreement / n,
            "avg_confidence": total_confidence / n,
            "signal_distribution": signals,
            "avg_participating_agents": sum(r.participating_agents for r in self.consensus_history) / n,
            "debate_triggered_pct": sum(1 for r in self.consensus_history if r.requires_debate) / n,
        }

    def clear_history(self) -> None:
        """Clear consensus history."""
        self.consensus_history = []


def create_multi_agent_consensus(
    custom_weights: dict[AgentType, float] | None = None,
    strong_signal_threshold: float = 0.7,
    debate_trigger_threshold: float = 0.4,
    use_confidence_weighting: bool = True,
) -> MultiAgentConsensus:
    """
    Factory function to create multi-agent consensus mechanism.

    Args:
        custom_weights: Custom weights for agent types
        strong_signal_threshold: Threshold for strong signals
        debate_trigger_threshold: Agreement below this triggers debate
        use_confidence_weighting: Whether to weight by agent confidence

    Returns:
        Configured MultiAgentConsensus instance
    """
    config = ConsensusConfig(
        strong_signal_threshold=strong_signal_threshold,
        debate_trigger_threshold=debate_trigger_threshold,
        use_confidence_weighting=use_confidence_weighting,
    )
    return MultiAgentConsensus(
        config=config,
        custom_weights=custom_weights,
    )


def opinion_from_agent_response(
    agent_response: Any,
    agent_type: AgentType,
) -> AgentOpinion | None:
    """
    Convert an AgentResponse to an AgentOpinion for consensus.

    Args:
        agent_response: AgentResponse from any trading agent
        agent_type: Type of the agent

    Returns:
        AgentOpinion if conversion successful, None otherwise
    """
    try:
        # Try to parse JSON response
        try:
            data = json.loads(agent_response.final_answer)
        except (json.JSONDecodeError, AttributeError):
            data = {}

        # Extract signal score from various possible fields
        signal_score = data.get("signal_score", 0.0)
        if signal_score == 0.0:
            # Try to infer from sentiment/direction
            sentiment = data.get("sentiment", data.get("direction", "neutral"))
            if isinstance(sentiment, str):
                if sentiment.lower() in ["bullish", "positive", "buy"]:
                    signal_score = 0.5
                elif sentiment.lower() in ["bearish", "negative", "sell"]:
                    signal_score = -0.5

        return AgentOpinion(
            agent_type=agent_type,
            agent_name=getattr(agent_response, "agent_name", "unknown"),
            signal_score=signal_score,
            confidence=getattr(agent_response, "confidence", 0.5),
            reasoning=data.get("reasoning", data.get("analysis", "")),
            key_factors=data.get("key_factors", data.get("key_points", [])),
            risk_factors=data.get("risk_factors", data.get("risks", [])),
            metadata={"raw_response": agent_response.final_answer[:500]},
        )

    except Exception:
        return None
