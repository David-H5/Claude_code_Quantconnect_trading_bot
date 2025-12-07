"""
Integration tests for Sprint 1.6 Unified Decision Context.

Tests the DecisionContextManager and UnifiedDecisionContext linking:
- DecisionLogger
- ReasoningLogger
- AnomalyDetector
- Explainer data

Part of UPGRADE-010: Advanced AI Features
Phase: Sprint 1.6 (Gap Resolution)
"""

from datetime import datetime, timezone

from evaluation.decision_context import (
    ContextCompleteness,
    DecisionContextBuilder,
    DecisionContextManager,
    ExplanationContext,
    MarketContext,
    UnifiedDecisionContext,
    create_context_manager,
    create_decision_context,
)
from llm.decision_logger import (
    AgentDecisionLog,
    Alternative,
    DecisionLogger,
    DecisionOutcome,
    DecisionType,
    ReasoningStep,
    RiskAssessment,
    RiskLevel,
)
from llm.reasoning_logger import ReasoningLogger
from models.anomaly_detector import AnomalyResult, AnomalySeverity, AnomalyType


def _now() -> datetime:
    """Get current time in UTC."""
    return datetime.now(timezone.utc)


def _create_test_decision(
    log_id: str = "test-log-1",
    agent_name: str = "test_agent",
    confidence: float = 0.85,
) -> AgentDecisionLog:
    """Create a test decision log with required fields."""
    return AgentDecisionLog(
        log_id=log_id,
        timestamp=_now(),
        agent_name=agent_name,
        agent_role="analyst",
        decision_type=DecisionType.TRADE,
        decision="BUY SPY",
        confidence=confidence,
        context={"symbol": "SPY"},
        query="Analyze SPY",
        reasoning_chain=[ReasoningStep(1, "Price rising", 0.8)],
        final_reasoning="Based on technical analysis",
        alternatives_considered=[Alternative("HOLD", "Wait for better entry", 0.3)],
        risk_assessment=RiskAssessment(
            overall_level=RiskLevel.MEDIUM,
            factors=["volatility"],
            mitigation_steps=["stop loss"],
            worst_case_scenario="5% loss",
        ),
        execution_time_ms=150.0,
        outcome=DecisionOutcome.PENDING,
    )


def _create_test_anomaly(
    anomaly_type: AnomalyType = AnomalyType.VOLATILITY_SPIKE,
    severity: AnomalySeverity = AnomalySeverity.HIGH,
) -> AnomalyResult:
    """Create a test anomaly result with required fields."""
    return AnomalyResult(
        is_anomaly=True,
        anomaly_type=anomaly_type,
        severity=severity,
        score=0.85,
        threshold=0.5,
        timestamp=_now(),
        feature_values={"volatility": 0.05},
        description="Volatility spike detected",
        recommended_action="reduce position size",
    )


class TestDecisionContextBuilder:
    """Tests for DecisionContextBuilder."""

    def test_build_empty_context(self):
        """Test building context with no components."""
        builder = DecisionContextBuilder()
        context = builder.build()

        assert context.context_id is not None
        assert context.created_at is not None
        assert context.completeness == ContextCompleteness.EMPTY
        assert context.decision is None
        assert context.reasoning_chain is None
        assert context.market_context is None

    def test_build_with_decision(self):
        """Test building context with decision log."""
        decision = _create_test_decision()

        builder = DecisionContextBuilder()
        context = builder.with_decision(decision).build()

        assert context.decision == decision
        assert context.agent_name == "test_agent"
        assert context.completeness == ContextCompleteness.MINIMAL

    def test_build_with_anomalies(self):
        """Test building context with anomalies."""
        anomalies = [_create_test_anomaly()]

        builder = DecisionContextBuilder()
        context = builder.with_anomalies(anomalies).build()

        assert context.market_context is not None
        assert context.market_context.anomaly_count == 1
        assert context.market_context.has_critical_anomaly is True
        assert len(context.market_context.anomalies) == 1

    def test_build_with_explanation(self):
        """Test building context with explanation data."""
        builder = DecisionContextBuilder()
        context = builder.with_explanation_data(
            explanation_type="shap",
            top_features=[{"name": "rsi", "contribution": 0.3}],
            feature_contributions={"rsi": 0.3, "macd": 0.2},
            model_confidence=0.88,
        ).build()

        assert context.explanation is not None
        assert context.explanation.explanation_type == "shap"
        assert context.explanation.model_confidence == 0.88
        assert len(context.explanation.top_features) == 1

    def test_build_full_context(self):
        """Test building context with all components."""
        decision = _create_test_decision()
        anomalies = [
            _create_test_anomaly(
                anomaly_type=AnomalyType.VOLUME_SPIKE,
                severity=AnomalySeverity.MEDIUM,
            )
        ]

        builder = DecisionContextBuilder()
        context = (
            builder.with_decision(decision)
            .with_anomalies(anomalies)
            .with_explanation_data(
                explanation_type="shap",
                top_features=[],
                feature_contributions={},
                model_confidence=0.9,
            )
            .with_symbol("SPY")
            .build()
        )

        # Without reasoning chain, only 3 components
        assert context.completeness == ContextCompleteness.PARTIAL
        assert context.symbol == "SPY"


class TestUnifiedDecisionContext:
    """Tests for UnifiedDecisionContext."""

    def test_calculate_completeness_full(self):
        """Test completeness calculation with all components."""
        context = UnifiedDecisionContext(
            context_id="test-1",
            created_at=_now(),
        )

        # Add decision
        context.decision = _create_test_decision()

        # Add reasoning chain
        from llm.reasoning_logger import ReasoningChain

        context.reasoning_chain = ReasoningChain(
            chain_id="chain-1",
            agent_name="agent",
            task="analysis",
            started_at=_now(),
            steps=[ReasoningStep(step_number=1, thought="Step", confidence=0.8)],
            final_confidence=0.8,
        )

        # Add market context
        context.market_context = MarketContext(
            timestamp=_now(),
            anomaly_count=1,
        )

        # Add explanation
        context.explanation = ExplanationContext(
            explanation_type="shap",
            model_confidence=0.85,
        )

        completeness = context.calculate_completeness()
        assert completeness == ContextCompleteness.FULL

    def test_get_confidence_score(self):
        """Test confidence score calculation."""
        context = UnifiedDecisionContext(
            context_id="test-1",
            created_at=_now(),
        )

        context.decision = _create_test_decision(confidence=0.8)
        context.explanation = ExplanationContext(
            explanation_type="shap",
            model_confidence=0.9,
        )

        # Weighted avg: (0.8 * 0.4 + 0.9 * 0.3) / (0.4 + 0.3) = 0.59 / 0.7 = 0.843
        score = context.get_confidence_score()
        assert 0.84 <= score <= 0.86

    def test_has_anomaly_warnings(self):
        """Test anomaly warning detection."""
        context = UnifiedDecisionContext(
            context_id="test-1",
            created_at=_now(),
        )

        assert context.has_anomaly_warnings() is False

        context.market_context = MarketContext(
            timestamp=_now(),
            has_critical_anomaly=True,
        )

        assert context.has_anomaly_warnings() is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        context = UnifiedDecisionContext(
            context_id="test-123",
            created_at=_now(),
            agent_name="test_agent",
            symbol="AAPL",
            completeness=ContextCompleteness.MINIMAL,
        )

        result = context.to_dict()

        assert result["context_id"] == "test-123"
        assert result["agent_name"] == "test_agent"
        assert result["symbol"] == "AAPL"
        assert result["completeness"] == "minimal"
        assert "created_at" in result


class TestDecisionContextManager:
    """Tests for DecisionContextManager."""

    def test_create_context_without_components(self):
        """Test creating context without component loggers."""
        manager = DecisionContextManager()
        context = manager.create_context()

        assert context is not None
        assert context.context_id is not None
        assert context.completeness == ContextCompleteness.EMPTY

    def test_create_context_with_decision(self):
        """Test creating context with decision log."""
        decision = _create_test_decision()

        manager = DecisionContextManager()
        context = manager.create_context(decision=decision)

        assert context.decision == decision
        assert context.agent_name == "test_agent"

    def test_get_context_by_id(self):
        """Test retrieving context by ID."""
        manager = DecisionContextManager()
        context = manager.create_context()

        retrieved = manager.get_context(context.context_id)
        assert retrieved is not None
        assert retrieved.context_id == context.context_id

    def test_get_context_by_decision_id(self):
        """Test retrieving context by decision ID."""
        decision = _create_test_decision(log_id="unique-log-id")

        manager = DecisionContextManager()
        context = manager.create_context(decision=decision)

        retrieved = manager.get_context_by_decision("unique-log-id")
        assert retrieved is not None
        assert retrieved.decision.log_id == "unique-log-id"

    def test_get_recent_contexts(self):
        """Test getting recent contexts."""
        manager = DecisionContextManager()

        # Create multiple contexts
        for i in range(5):
            decision = _create_test_decision(
                log_id=f"log-{i}",
                agent_name=f"agent_{i % 2}",
            )
            manager.create_context(decision=decision)

        recent = manager.get_recent_contexts(limit=3)
        assert len(recent) == 3

        # Filter by agent
        agent_0_contexts = manager.get_recent_contexts(agent_name="agent_0")
        assert all(c.agent_name == "agent_0" for c in agent_0_contexts)

    def test_callback_on_context_created(self):
        """Test callback is called when context is created."""
        manager = DecisionContextManager()
        callback_called = []

        def on_context(ctx: UnifiedDecisionContext):
            callback_called.append(ctx.context_id)

        manager.register_callback(on_context)
        context = manager.create_context()

        assert len(callback_called) == 1
        assert callback_called[0] == context.context_id

    def test_get_statistics(self):
        """Test getting manager statistics."""
        manager = DecisionContextManager()

        # Create contexts with different configurations
        manager.create_context()
        decision = _create_test_decision(agent_name="agent_a", confidence=0.9)
        manager.create_context(decision=decision)

        stats = manager.get_statistics()

        assert stats["total_contexts"] == 2
        assert "completeness_distribution" in stats
        assert "average_confidence" in stats
        assert "contexts_by_agent" in stats

    def test_max_contexts_limit(self):
        """Test that max contexts limit is enforced."""
        manager = DecisionContextManager(max_contexts=5)

        # Create more than max contexts
        for i in range(10):
            manager.create_context()

        # Should only have max_contexts stored
        stats = manager.get_statistics()
        assert stats["total_contexts"] == 5


class TestDecisionContextFactoryFunctions:
    """Tests for factory functions."""

    def test_create_decision_context(self):
        """Test create_decision_context factory function."""
        context = create_decision_context(
            agent_name="test_agent",
            symbol="SPY",
            explanation_type="shap",
            top_features=[{"name": "rsi", "contribution": 0.3}],
            feature_contributions={"rsi": 0.3},
            model_confidence=0.85,
        )

        assert context.agent_name == "test_agent"
        assert context.symbol == "SPY"
        assert context.explanation is not None
        assert context.explanation.explanation_type == "shap"

    def test_create_context_manager(self):
        """Test create_context_manager factory function."""
        decision_logger = DecisionLogger()
        reasoning_logger = ReasoningLogger()

        manager = create_context_manager(
            decision_logger=decision_logger,
            reasoning_logger=reasoning_logger,
            max_contexts=100,
        )

        assert manager is not None
        assert manager._max_contexts == 100


class TestIntegrationWithSprint1Components:
    """Integration tests with actual Sprint 1 components."""

    def test_full_integration_flow(self):
        """Test complete integration with all Sprint 1 components."""
        # Create Sprint 1 components
        decision_logger = DecisionLogger()
        reasoning_logger = ReasoningLogger()

        # Create a reasoning chain
        chain = reasoning_logger.start_chain(
            agent_name="technical_analyst",
            task="market_analysis",
        )
        # Complete chain (steps are added internally or via chain object)
        reasoning_logger.complete_chain(
            chain.chain_id,
            decision="BUY SPY",
            confidence=0.82,
        )

        # Create a decision linked to the chain
        decision = decision_logger.log_decision(
            agent_name="technical_analyst",
            agent_role="analyst",
            decision_type=DecisionType.TRADE,
            decision="BUY SPY at 450.00",
            confidence=0.82,
            context={"symbol": "SPY", "price": 450.0},
            query="Should we buy SPY?",
            reasoning_chain=[
                ReasoningStep(1, "RSI oversold", 0.85),
                ReasoningStep(2, "MACD bullish", 0.80),
            ],
            final_reasoning="Technical indicators show bullish momentum",
            alternatives=[Alternative("HOLD", "Wait for confirmation", 0.3)],
            risk_assessment=RiskAssessment(
                overall_level=RiskLevel.MEDIUM,
                factors=["market volatility"],
                mitigation_steps=["stop loss at 445"],
                worst_case_scenario="5% loss",
            ),
            execution_time_ms=125.0,
            reasoning_chain_id=chain.chain_id,
        )

        # Create context manager with components
        manager = create_context_manager(
            decision_logger=decision_logger,
            reasoning_logger=reasoning_logger,
        )

        # Create unified context
        context = manager.create_context(
            decision=decision,
            include_anomalies=False,  # No anomaly detector in this test
        )

        # Verify context links everything
        assert context.decision is not None
        assert context.decision.log_id == decision.log_id
        assert context.reasoning_chain is not None
        assert context.reasoning_chain.chain_id == chain.chain_id
        assert context.agent_name == "technical_analyst"

        # Verify confidence scoring
        confidence = context.get_confidence_score()
        assert 0.5 <= confidence <= 1.0

        # Verify serialization
        ctx_dict = context.to_dict()
        assert ctx_dict["decision"] is not None
        assert ctx_dict["reasoning_chain"] is not None
        assert "confidence_score" in ctx_dict

    def test_context_retrieval_by_chain_id(self):
        """Test retrieving context by reasoning chain ID."""
        reasoning_logger = ReasoningLogger()
        chain = reasoning_logger.start_chain("agent", "analysis")
        reasoning_logger.complete_chain(chain.chain_id, "HOLD", 0.8)

        manager = create_context_manager(reasoning_logger=reasoning_logger)

        # Create context with chain
        builder = DecisionContextBuilder()
        builder.with_reasoning_chain(chain)
        context = builder.build()

        # Store it manually to test retrieval
        manager._contexts[context.context_id] = context
        manager._contexts_by_chain[chain.chain_id] = context.context_id

        # Retrieve by chain ID
        retrieved = manager.get_context_by_chain(chain.chain_id)
        assert retrieved is not None
        assert retrieved.reasoning_chain.chain_id == chain.chain_id


class TestExportFromEvaluationModule:
    """Test that all exports work from evaluation module."""

    def test_import_from_evaluation(self):
        """Test importing decision context components from evaluation."""
        from evaluation import (
            ContextCompleteness,
            DecisionContextBuilder,
            DecisionContextManager,
            ExplanationContext,
            MarketContext,
            UnifiedDecisionContext,
            create_context_manager,
            create_decision_context,
        )

        # Verify all imports work
        assert ContextCompleteness.FULL.value == "full"
        assert UnifiedDecisionContext is not None
        assert DecisionContextBuilder is not None
        assert DecisionContextManager is not None
        assert MarketContext is not None
        assert ExplanationContext is not None
        assert callable(create_decision_context)
        assert callable(create_context_manager)
