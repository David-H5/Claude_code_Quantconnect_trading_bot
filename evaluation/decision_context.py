"""
Unified Decision Context (Sprint 1.6 - Gap Resolution)

Provides a unified context linking all Sprint 1 components:
- DecisionLogger: Decision audit trails
- ReasoningLogger: Chain-of-thought reasoning
- AnomalyDetector: Market anomaly detection
- Explainer: SHAP/LIME explanations

This module bridges the gap between individual Sprint 1 components,
enabling full traceability from decision → reasoning → context → explanation.

Part of UPGRADE-010: Advanced AI Features
Phase: Sprint 1.6 (Gap Resolution)

QuantConnect Compatible: Yes
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from llm.decision_logger import AgentDecisionLog, DecisionLogger
from llm.reasoning_logger import ReasoningChain, ReasoningLogger
from models.anomaly_detector import AnomalyDetector, AnomalyResult


class ContextCompleteness(Enum):
    """Level of context completeness."""

    FULL = "full"  # All 4 components linked
    PARTIAL = "partial"  # 2-3 components linked
    MINIMAL = "minimal"  # Only decision present
    EMPTY = "empty"  # No components


@dataclass
class MarketContext:
    """Market conditions at decision time."""

    timestamp: datetime
    anomalies: list[AnomalyResult] = field(default_factory=list)
    anomaly_count: int = 0
    has_critical_anomaly: bool = False
    recent_volatility: float | None = None
    market_regime: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "anomalies": [a.to_dict() for a in self.anomalies],
            "anomaly_count": self.anomaly_count,
            "has_critical_anomaly": self.has_critical_anomaly,
            "recent_volatility": self.recent_volatility,
            "market_regime": self.market_regime,
        }


@dataclass
class ExplanationContext:
    """Explanation data for the decision."""

    explanation_type: str = "none"
    top_features: list[dict[str, Any]] = field(default_factory=list)
    feature_contributions: dict[str, float] = field(default_factory=dict)
    model_confidence: float = 0.0
    explanation_timestamp: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "explanation_type": self.explanation_type,
            "top_features": self.top_features,
            "feature_contributions": self.feature_contributions,
            "model_confidence": self.model_confidence,
            "explanation_timestamp": (self.explanation_timestamp.isoformat() if self.explanation_timestamp else None),
        }


@dataclass
class UnifiedDecisionContext:
    """
    Unified context linking all Sprint 1 components for a single decision.

    Sprint 1.6: Bridges the gap between individual components,
    enabling full traceability from decision → reasoning → context → explanation.
    """

    # Core identifiers
    context_id: str
    created_at: datetime

    # Decision component (from DecisionLogger)
    decision: AgentDecisionLog | None = None

    # Reasoning component (from ReasoningLogger)
    reasoning_chain: ReasoningChain | None = None

    # Market context (from AnomalyDetector)
    market_context: MarketContext | None = None

    # Explanation context (from Explainer)
    explanation: ExplanationContext | None = None

    # Metadata
    agent_name: str = ""
    symbol: str = ""
    completeness: ContextCompleteness = ContextCompleteness.EMPTY

    def calculate_completeness(self) -> ContextCompleteness:
        """Calculate context completeness based on linked components."""
        component_count = sum(
            [
                self.decision is not None,
                self.reasoning_chain is not None,
                self.market_context is not None and self.market_context.anomaly_count >= 0,
                self.explanation is not None and self.explanation.explanation_type != "none",
            ]
        )

        if component_count == 4:
            return ContextCompleteness.FULL
        elif component_count >= 2:
            return ContextCompleteness.PARTIAL
        elif component_count == 1:
            return ContextCompleteness.MINIMAL
        else:
            return ContextCompleteness.EMPTY

    def get_confidence_score(self) -> float:
        """Get combined confidence score from all components."""
        scores: list[float] = []
        weights: list[float] = []

        if self.decision:
            scores.append(self.decision.confidence)
            weights.append(0.4)  # Decision confidence weighted highest

        if self.reasoning_chain:
            scores.append(self.reasoning_chain.final_confidence)
            weights.append(0.3)  # Reasoning confidence

        if self.explanation and self.explanation.model_confidence > 0:
            scores.append(self.explanation.model_confidence)
            weights.append(0.3)  # Model confidence

        if not scores:
            return 0.0

        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        return sum(s * w for s, w in zip(scores, weights)) / total_weight

    def has_anomaly_warnings(self) -> bool:
        """Check if there are anomaly warnings in market context."""
        if not self.market_context:
            return False
        return self.market_context.has_critical_anomaly or self.market_context.anomaly_count > 0

    def get_summary(self) -> str:
        """Get a brief summary of the context."""
        parts = [f"Context {self.context_id[:8]}"]

        if self.agent_name:
            parts.append(f"Agent: {self.agent_name}")

        if self.decision:
            parts.append(f"Decision: {self.decision.decision_type.value}")

        if self.reasoning_chain:
            parts.append(f"Steps: {len(self.reasoning_chain.steps)}")

        if self.market_context:
            parts.append(f"Anomalies: {self.market_context.anomaly_count}")

        parts.append(f"Completeness: {self.completeness.value}")

        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context_id": self.context_id,
            "created_at": self.created_at.isoformat(),
            "agent_name": self.agent_name,
            "symbol": self.symbol,
            "completeness": self.completeness.value,
            "confidence_score": self.get_confidence_score(),
            "has_anomaly_warnings": self.has_anomaly_warnings(),
            "decision": self.decision.to_dict() if self.decision else None,
            "reasoning_chain": (self.reasoning_chain.to_dict() if self.reasoning_chain else None),
            "market_context": (self.market_context.to_dict() if self.market_context else None),
            "explanation": self.explanation.to_dict() if self.explanation else None,
        }


class DecisionContextBuilder:
    """
    Builder for creating UnifiedDecisionContext instances.

    Sprint 1.6: Provides fluent API for linking Sprint 1 components.
    """

    def __init__(self):
        """Initialize builder."""
        import hashlib

        timestamp = datetime.utcnow()
        content = f"decision_context:{timestamp.isoformat()}"
        self._context_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        self._created_at = timestamp
        self._decision: AgentDecisionLog | None = None
        self._reasoning_chain: ReasoningChain | None = None
        self._market_context: MarketContext | None = None
        self._explanation: ExplanationContext | None = None
        self._agent_name: str = ""
        self._symbol: str = ""

    def with_decision(self, decision: AgentDecisionLog) -> "DecisionContextBuilder":
        """Add decision log to context."""
        self._decision = decision
        if decision.agent_name:
            self._agent_name = decision.agent_name
        return self

    def with_reasoning_chain(self, chain: ReasoningChain) -> "DecisionContextBuilder":
        """Add reasoning chain to context."""
        self._reasoning_chain = chain
        if chain.agent_name and not self._agent_name:
            self._agent_name = chain.agent_name
        return self

    def with_market_context(self, context: MarketContext) -> "DecisionContextBuilder":
        """Add market context to context."""
        self._market_context = context
        return self

    def with_anomalies(
        self, anomalies: list[AnomalyResult], timestamp: datetime | None = None
    ) -> "DecisionContextBuilder":
        """Add anomalies as market context."""
        has_critical = any(a.severity.value in ["critical", "high"] for a in anomalies)
        self._market_context = MarketContext(
            timestamp=timestamp or datetime.utcnow(),
            anomalies=anomalies,
            anomaly_count=len(anomalies),
            has_critical_anomaly=has_critical,
        )
        return self

    def with_explanation(self, explanation: ExplanationContext) -> "DecisionContextBuilder":
        """Add explanation context."""
        self._explanation = explanation
        return self

    def with_explanation_data(
        self,
        explanation_type: str,
        top_features: list[dict[str, Any]],
        feature_contributions: dict[str, float],
        model_confidence: float = 0.0,
    ) -> "DecisionContextBuilder":
        """Add explanation data directly."""
        self._explanation = ExplanationContext(
            explanation_type=explanation_type,
            top_features=top_features,
            feature_contributions=feature_contributions,
            model_confidence=model_confidence,
            explanation_timestamp=datetime.utcnow(),
        )
        return self

    def with_symbol(self, symbol: str) -> "DecisionContextBuilder":
        """Set trading symbol."""
        self._symbol = symbol
        return self

    def with_agent(self, agent_name: str) -> "DecisionContextBuilder":
        """Set agent name."""
        self._agent_name = agent_name
        return self

    def build(self) -> UnifiedDecisionContext:
        """Build the unified decision context."""
        context = UnifiedDecisionContext(
            context_id=self._context_id,
            created_at=self._created_at,
            decision=self._decision,
            reasoning_chain=self._reasoning_chain,
            market_context=self._market_context,
            explanation=self._explanation,
            agent_name=self._agent_name,
            symbol=self._symbol,
        )
        context.completeness = context.calculate_completeness()
        return context


class DecisionContextManager:
    """
    Manager for creating and tracking unified decision contexts.

    Sprint 1.6: Provides integration with all Sprint 1 components.
    Sprint 1.7: Added Explainer integration for auto-explanation.
    """

    def __init__(
        self,
        decision_logger: DecisionLogger | None = None,
        reasoning_logger: ReasoningLogger | None = None,
        anomaly_detector: AnomalyDetector | None = None,
        explainer: Any | None = None,
        max_contexts: int = 1000,
    ):
        """
        Initialize the context manager.

        Args:
            decision_logger: Optional DecisionLogger instance
            reasoning_logger: Optional ReasoningLogger instance
            anomaly_detector: Optional AnomalyDetector instance
            explainer: Optional BaseExplainer instance (Sprint 1.7)
            max_contexts: Maximum contexts to retain in memory
        """
        self._decision_logger = decision_logger
        self._reasoning_logger = reasoning_logger
        self._anomaly_detector = anomaly_detector
        self._explainer = explainer  # Sprint 1.7
        self._max_contexts = max_contexts
        self._contexts: dict[str, UnifiedDecisionContext] = {}
        self._contexts_by_decision: dict[str, str] = {}  # decision_id -> context_id
        self._contexts_by_chain: dict[str, str] = {}  # chain_id -> context_id
        self._on_context_created: list[Callable[[UnifiedDecisionContext], None]] = []

    def register_callback(self, callback: Callable[[UnifiedDecisionContext], None]) -> None:
        """Register callback for new contexts."""
        self._on_context_created.append(callback)

    def _generate_explanation(self, feature_data: dict[str, float]) -> ExplanationContext | None:
        """
        Generate explanation using the explainer.

        Sprint 1.7: Auto-explanation integration.

        Args:
            feature_data: Dictionary of feature names to values

        Returns:
            ExplanationContext or None if generation fails
        """
        if not self._explainer:
            return None

        try:
            import numpy as np

            # Convert feature dict to array for explainer
            feature_names = list(feature_data.keys())
            feature_values = np.array([list(feature_data.values())])

            # Generate explanation
            explanation = self._explainer.explain(
                instance=feature_values[0],
                feature_names=feature_names,
            )

            # Convert to ExplanationContext
            top_features = []
            for contrib in explanation.feature_contributions[:5]:  # Top 5
                top_features.append(
                    {
                        "name": contrib.feature_name,
                        "contribution": contrib.contribution,
                        "value": contrib.feature_value,
                    }
                )

            feature_contributions = {c.feature_name: c.contribution for c in explanation.feature_contributions}

            return ExplanationContext(
                explanation_type=explanation.explanation_type.value,
                top_features=top_features,
                feature_contributions=feature_contributions,
                model_confidence=explanation.confidence,
                explanation_timestamp=datetime.utcnow(),
            )

        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Failed to generate explanation: {e}")
            return None

    def create_context(
        self,
        decision: AgentDecisionLog | None = None,
        reasoning_chain_id: str | None = None,
        include_anomalies: bool = True,
        anomaly_window_minutes: int = 5,
        auto_explain: bool = True,
        feature_data: dict[str, float] | None = None,
    ) -> UnifiedDecisionContext:
        """
        Create a unified decision context.

        Sprint 1.7: Added auto_explain parameter for automatic SHAP/LIME explanations.

        Args:
            decision: Optional AgentDecisionLog to include
            reasoning_chain_id: Optional chain_id to link reasoning
            include_anomalies: Whether to include recent anomalies
            anomaly_window_minutes: Time window for recent anomalies
            auto_explain: Whether to auto-generate explanation (Sprint 1.7)
            feature_data: Optional feature data for explanation generation

        Returns:
            UnifiedDecisionContext with linked components
        """
        builder = DecisionContextBuilder()

        # Add decision
        if decision:
            builder.with_decision(decision)
            if decision.reasoning_chain_id:
                reasoning_chain_id = decision.reasoning_chain_id

        # Add reasoning chain
        if reasoning_chain_id and self._reasoning_logger:
            chain = self._reasoning_logger.get_chain(reasoning_chain_id)
            if chain:
                builder.with_reasoning_chain(chain)

        # Add recent anomalies
        if include_anomalies and self._anomaly_detector:
            history = self._anomaly_detector.get_anomaly_history(limit=10)
            cutoff = datetime.utcnow()
            recent = [a for a in history if (cutoff - a.timestamp).total_seconds() < anomaly_window_minutes * 60]
            if recent:
                builder.with_anomalies(recent)

        # Sprint 1.7: Auto-generate explanation using explainer
        if auto_explain and self._explainer and feature_data:
            explanation = self._generate_explanation(feature_data)
            if explanation:
                builder.with_explanation(explanation)

        context = builder.build()

        # Store context
        self._store_context(context, decision)

        # Notify callbacks
        for callback in self._on_context_created:
            try:
                callback(context)
            except Exception:
                pass  # Don't let callback errors break context creation

        return context

    def _store_context(self, context: UnifiedDecisionContext, decision: AgentDecisionLog | None) -> None:
        """Store context and update indexes."""
        # Enforce max contexts
        if len(self._contexts) >= self._max_contexts:
            # Remove oldest context
            oldest_id = next(iter(self._contexts))
            del self._contexts[oldest_id]

        self._contexts[context.context_id] = context

        # Update indexes
        if decision:
            self._contexts_by_decision[decision.log_id] = context.context_id

        if context.reasoning_chain:
            self._contexts_by_chain[context.reasoning_chain.chain_id] = context.context_id

    def get_context(self, context_id: str) -> UnifiedDecisionContext | None:
        """Get context by ID."""
        return self._contexts.get(context_id)

    def get_context_by_decision(self, decision_id: str) -> UnifiedDecisionContext | None:
        """Get context by decision ID."""
        context_id = self._contexts_by_decision.get(decision_id)
        if context_id:
            return self._contexts.get(context_id)
        return None

    def get_context_by_chain(self, chain_id: str) -> UnifiedDecisionContext | None:
        """Get context by reasoning chain ID."""
        context_id = self._contexts_by_chain.get(chain_id)
        if context_id:
            return self._contexts.get(context_id)
        return None

    def get_recent_contexts(self, limit: int = 10, agent_name: str | None = None) -> list[UnifiedDecisionContext]:
        """Get recent contexts, optionally filtered by agent."""
        contexts = list(self._contexts.values())

        if agent_name:
            contexts = [c for c in contexts if c.agent_name == agent_name]

        # Sort by created_at descending
        contexts.sort(key=lambda c: c.created_at, reverse=True)

        return contexts[:limit]

    def get_contexts_with_anomalies(self) -> list[UnifiedDecisionContext]:
        """Get all contexts that have anomaly warnings."""
        return [c for c in self._contexts.values() if c.has_anomaly_warnings()]

    def get_statistics(self) -> dict[str, Any]:
        """Get manager statistics."""
        contexts = list(self._contexts.values())

        if not contexts:
            return {
                "total_contexts": 0,
                "completeness_distribution": {},
                "average_confidence": 0.0,
                "contexts_with_anomalies": 0,
            }

        completeness_dist: dict[str, int] = {}
        for c in contexts:
            key = c.completeness.value
            completeness_dist[key] = completeness_dist.get(key, 0) + 1

        avg_confidence = sum(c.get_confidence_score() for c in contexts) / len(contexts)
        anomaly_count = sum(1 for c in contexts if c.has_anomaly_warnings())

        return {
            "total_contexts": len(contexts),
            "completeness_distribution": completeness_dist,
            "average_confidence": avg_confidence,
            "contexts_with_anomalies": anomaly_count,
            "contexts_by_agent": self._get_agent_distribution(contexts),
        }

    def _get_agent_distribution(self, contexts: list[UnifiedDecisionContext]) -> dict[str, int]:
        """Get distribution of contexts by agent."""
        dist: dict[str, int] = {}
        for c in contexts:
            agent = c.agent_name or "unknown"
            dist[agent] = dist.get(agent, 0) + 1
        return dist


def create_decision_context(
    decision: AgentDecisionLog | None = None,
    reasoning_chain: ReasoningChain | None = None,
    anomalies: list[AnomalyResult] | None = None,
    explanation_type: str = "none",
    top_features: list[dict[str, Any]] | None = None,
    feature_contributions: dict[str, float] | None = None,
    model_confidence: float = 0.0,
    agent_name: str = "",
    symbol: str = "",
) -> UnifiedDecisionContext:
    """
    Factory function to create a unified decision context.

    Sprint 1.6: Convenience function for creating contexts without manager.

    Args:
        decision: Optional AgentDecisionLog
        reasoning_chain: Optional ReasoningChain
        anomalies: Optional list of AnomalyResult objects
        explanation_type: Type of explanation (shap, lime, feature_importance, none)
        top_features: Top contributing features
        feature_contributions: Feature contribution mapping
        model_confidence: Model confidence score
        agent_name: Name of the agent
        symbol: Trading symbol

    Returns:
        UnifiedDecisionContext with provided components
    """
    builder = DecisionContextBuilder()

    if decision:
        builder.with_decision(decision)

    if reasoning_chain:
        builder.with_reasoning_chain(reasoning_chain)

    if anomalies:
        builder.with_anomalies(anomalies)

    if explanation_type != "none":
        builder.with_explanation_data(
            explanation_type=explanation_type,
            top_features=top_features or [],
            feature_contributions=feature_contributions or {},
            model_confidence=model_confidence,
        )

    if agent_name:
        builder.with_agent(agent_name)

    if symbol:
        builder.with_symbol(symbol)

    return builder.build()


def create_context_manager(
    decision_logger: DecisionLogger | None = None,
    reasoning_logger: ReasoningLogger | None = None,
    anomaly_detector: AnomalyDetector | None = None,
    explainer: Any | None = None,
    max_contexts: int = 1000,
) -> DecisionContextManager:
    """
    Factory function to create a context manager.

    Sprint 1.7: Added explainer parameter for auto-explanation.

    Args:
        decision_logger: Optional DecisionLogger instance
        reasoning_logger: Optional ReasoningLogger instance
        anomaly_detector: Optional AnomalyDetector instance
        explainer: Optional BaseExplainer instance (Sprint 1.7)
        max_contexts: Maximum contexts to retain

    Returns:
        Configured DecisionContextManager
    """
    return DecisionContextManager(
        decision_logger=decision_logger,
        reasoning_logger=reasoning_logger,
        anomaly_detector=anomaly_detector,
        explainer=explainer,
        max_contexts=max_contexts,
    )
