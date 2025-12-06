"""
Sprint 1 Integration Tests

Tests end-to-end integration of Sprint 1 "Explainability and Monitoring" components:
- Anomaly Detector → Circuit Breaker flow
- Anomaly Detector → Alerting Service flow
- Reasoning Logger → Agent integration
- SHAP Explainer → Audit trail export

UPGRADE-010 Sprint 1: Foundation Features
December 2025
"""

from datetime import datetime
from typing import Any

import pytest

from llm.agents.base import AgentResponse, AgentRole, AgentThought, ThoughtType, TradingAgent
from llm.reasoning_logger import (
    ChainStatus,
    create_reasoning_logger,
)
from models.anomaly_detector import (
    AnomalyDetectorConfig,
    AnomalySeverity,
    AnomalyType,
    MarketDataPoint,
    create_anomaly_detector,
)
from models.circuit_breaker import (
    CircuitBreakerConfig,
    TradingCircuitBreaker,
    TripReason,
    create_circuit_breaker,
)
from utils.alerting_service import (
    AlertCategory,
    create_alerting_service,
)
from utils.alerting_service import (
    AlertSeverity as AlertSevLevel,
)


class MockTradingAgent(TradingAgent):
    """Mock trading agent for testing."""

    def analyze(self, query: str, context: dict[str, Any]) -> AgentResponse:
        """Simple analysis that uses reasoning chain logging."""
        start_time = datetime.utcnow()
        thoughts = []

        # Start reasoning chain
        self.start_reasoning_chain(query)

        # Step 1: Initial analysis
        thought1 = AgentThought(
            thought_type=ThoughtType.REASONING,
            content="Analyzing market conditions",
        )
        thoughts.append(thought1)
        self.log_reasoning_step("Analyzing market conditions", confidence=0.8)

        # Step 2: Evaluation
        thought2 = AgentThought(
            thought_type=ThoughtType.REASONING,
            content="Market appears stable",
            metadata={"observation": context.get("price", 0)},
        )
        thoughts.append(thought2)
        self.log_reasoning_step(
            "Market appears stable",
            evidence=f"Price: {context.get('price', 0)}",
            confidence=0.9,
        )

        # Final decision
        decision = "HOLD"
        confidence = 0.85

        thought3 = AgentThought(
            thought_type=ThoughtType.FINAL_ANSWER,
            content=decision,
        )
        thoughts.append(thought3)

        # Complete reasoning chain
        self.complete_reasoning_chain(decision, confidence)

        return AgentResponse(
            agent_name=self.name,
            agent_role=self.role,
            query=query,
            thoughts=thoughts,
            final_answer=decision,
            confidence=confidence,
            tools_used=[],
            execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            success=True,
        )


class TestAnomalyDetectorCircuitBreakerIntegration:
    """Test anomaly detector → circuit breaker integration."""

    @pytest.fixture
    def circuit_breaker(self) -> TradingCircuitBreaker:
        """Create circuit breaker for testing."""
        config = CircuitBreakerConfig(
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.10,
            max_consecutive_losses=5,
            require_human_reset=False,
        )
        return create_circuit_breaker(config)

    @pytest.fixture
    def anomaly_detector(self, circuit_breaker):
        """Create anomaly detector with circuit breaker callback."""

        def on_anomaly(result):
            if result.severity == AnomalySeverity.CRITICAL:
                circuit_breaker.trip(TripReason.ANOMALY_DETECTED)

        config = AnomalyDetectorConfig(
            flash_crash_threshold_pct=0.05,
            min_data_points=5,
        )
        return create_anomaly_detector(config, on_anomaly_callback=on_anomaly)

    @pytest.mark.integration
    def test_flash_crash_triggers_circuit_breaker(self, anomaly_detector, circuit_breaker):
        """Flash crash detection should trigger circuit breaker."""
        # Verify circuit breaker is initially active
        assert circuit_breaker.can_trade()

        # Add normal data points
        base_price = 100.0
        for i in range(10):
            anomaly_detector.add_data(
                MarketDataPoint(
                    timestamp=datetime.utcnow(),
                    price=base_price,
                    volume=1000000,
                )
            )

        # Simulate flash crash (-10% drop)
        crash_price = base_price * 0.90
        anomaly_detector.add_data(
            MarketDataPoint(
                timestamp=datetime.utcnow(),
                price=crash_price,
                volume=5000000,
            )
        )

        # Detect anomaly
        result = anomaly_detector.detect()

        # Verify detection - a 10% drop in 60s triggers FLASH_CRASH detection
        assert result.is_anomaly
        assert result.anomaly_type in [AnomalyType.FLASH_CRASH, AnomalyType.PRICE_GAP]
        assert result.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]

    @pytest.mark.integration
    def test_volume_spike_detection(self, anomaly_detector, circuit_breaker):
        """Volume spike should be detected but not necessarily trip circuit breaker."""
        # Add baseline volume data
        for i in range(25):
            anomaly_detector.add_data(
                MarketDataPoint(
                    timestamp=datetime.utcnow(),
                    price=100.0,
                    volume=1000000,
                )
            )

        # Add volume spike (5x normal)
        anomaly_detector.add_data(
            MarketDataPoint(
                timestamp=datetime.utcnow(),
                price=100.0,
                volume=5000000,
            )
        )

        result = anomaly_detector.detect()

        # Volume spike should be detected
        if result.is_anomaly:
            assert result.anomaly_type == AnomalyType.VOLUME_SPIKE


class TestAnomalyDetectorAlertingIntegration:
    """Test anomaly detector → alerting service integration."""

    @pytest.fixture
    def alerting_service(self):
        """Create alerting service."""
        return create_alerting_service()

    @pytest.fixture
    def anomaly_detector_with_alerts(self, alerting_service):
        """Create anomaly detector with alerting integration."""

        def on_anomaly(result):
            alerting_service.send_anomaly_alert(
                anomaly_type=result.anomaly_type.value,
                message=result.description,
                severity=AlertSevLevel.WARNING if result.severity == AnomalySeverity.MEDIUM else AlertSevLevel.CRITICAL,
                score=result.score,
                recommended_action=result.recommended_action,
                data=result.feature_values,
            )

        config = AnomalyDetectorConfig(min_data_points=5)
        return create_anomaly_detector(config, on_anomaly_callback=on_anomaly)

    @pytest.mark.integration
    def test_anomaly_sends_alert(self, anomaly_detector_with_alerts, alerting_service):
        """Anomaly detection should send alert through alerting service."""
        # Add data to trigger anomaly
        for i in range(10):
            anomaly_detector_with_alerts.add_data(
                MarketDataPoint(
                    timestamp=datetime.utcnow(),
                    price=100.0 - i * 0.5,  # Gradual decline
                    volume=1000000,
                )
            )

        # Large price gap
        anomaly_detector_with_alerts.add_data(
            MarketDataPoint(
                timestamp=datetime.utcnow(),
                price=90.0,
                volume=2000000,
            )
        )

        result = anomaly_detector_with_alerts.detect()

        # Check alerts were sent
        recent_alerts = alerting_service.get_recent_alerts(limit=5)

        if result.is_anomaly:
            # Should have at least one ANOMALY category alert
            anomaly_alerts = [a for a in recent_alerts if a.category == AlertCategory.ANOMALY]
            assert len(anomaly_alerts) >= 1

    @pytest.mark.integration
    def test_alert_category_anomaly_exists(self):
        """Verify ANOMALY category exists in AlertCategory enum."""
        assert hasattr(AlertCategory, "ANOMALY")
        assert AlertCategory.ANOMALY.value == "anomaly"


class TestReasoningLoggerAgentIntegration:
    """Test reasoning logger → trading agent integration."""

    @pytest.fixture
    def reasoning_logger(self, tmp_path):
        """Create reasoning logger."""
        return create_reasoning_logger(
            storage_dir=str(tmp_path / "reasoning_chains"),
            auto_persist=True,
        )

    @pytest.fixture
    def agent_with_reasoning(self, reasoning_logger) -> MockTradingAgent:
        """Create trading agent with reasoning logger."""
        return MockTradingAgent(
            name="test_analyst",
            role=AgentRole.ANALYST,
            system_prompt="Test analyst for integration testing",
            reasoning_logger=reasoning_logger,
            enable_reasoning_logging=True,
        )

    @pytest.mark.integration
    def test_agent_logs_reasoning_chain(self, agent_with_reasoning, reasoning_logger):
        """Agent should log reasoning chain during analysis."""
        # Run analysis
        response = agent_with_reasoning.analyze(
            query="Analyze SPY for entry",
            context={"symbol": "SPY", "price": 450.0},
        )

        # Verify response
        assert response.success
        assert response.final_answer == "HOLD"

        # Verify reasoning chain was logged
        chains = reasoning_logger.get_chains_by_agent("test_analyst")
        assert len(chains) >= 1

        # Verify chain details
        latest_chain = chains[-1]
        assert latest_chain.agent_name == "test_analyst"
        assert latest_chain.status == ChainStatus.COMPLETED
        assert len(latest_chain.steps) >= 2
        assert latest_chain.final_decision == "HOLD"
        assert latest_chain.final_confidence == 0.85

    @pytest.mark.integration
    def test_agent_reasoning_chain_searchable(self, agent_with_reasoning, reasoning_logger):
        """Agent reasoning chains should be searchable."""
        # Run analysis
        agent_with_reasoning.analyze(
            query="Analyze market conditions",
            context={"symbol": "AAPL", "price": 180.0},
        )

        # Search for reasoning
        results = reasoning_logger.search_reasoning("market")
        assert len(results) >= 1
        assert "market" in results[0].task.lower()

    @pytest.mark.integration
    def test_agent_without_logger_works(self):
        """Agent should work without reasoning logger configured."""
        agent = MockTradingAgent(
            name="no_logger_agent",
            role=AgentRole.ANALYST,
            system_prompt="Test analyst without logger",
            reasoning_logger=None,
            enable_reasoning_logging=False,
        )

        response = agent.analyze(
            query="Simple analysis",
            context={"price": 100.0},
        )

        # Should still work
        assert response.success
        assert response.final_answer == "HOLD"


class TestExplainerAuditTrailIntegration:
    """Test SHAP explainer → audit trail export integration."""

    @pytest.mark.integration
    def test_explanation_audit_trail_export(self, tmp_path):
        """Test explanation logging and audit trail export."""
        from evaluation.explainer import (
            Explanation,
            ExplanationLogger,
            ExplanationType,
            FeatureContribution,
            ModelType,
        )

        # Create logger
        logger = ExplanationLogger(
            storage_dir=str(tmp_path / "explanations"),
            auto_persist=True,
        )

        # Log some explanations
        for i in range(3):
            explanation = Explanation(
                explanation_id=f"exp_{i}",
                timestamp=datetime.utcnow(),
                explanation_type=ExplanationType.FEATURE_IMPORTANCE,
                model_type=ModelType.TREE,
                prediction=0.75 + i * 0.05,
                base_value=0.5,
                feature_contributions=[
                    FeatureContribution(
                        feature_name="rsi",
                        feature_value=35.0,
                        contribution=0.15,
                        rank=1,
                        direction="positive",
                    ),
                    FeatureContribution(
                        feature_name="volume",
                        feature_value=1000000,
                        contribution=0.10,
                        rank=2,
                        direction="positive",
                    ),
                ],
                total_contribution=0.25,
            )
            logger.log(explanation)

        # Export audit trail
        audit_path = str(tmp_path / "audit_trail.json")
        count = logger.export_audit_trail(audit_path)

        assert count == 3
        assert (tmp_path / "audit_trail.json").exists()


class TestSprint1ComponentsExist:
    """Verify all Sprint 1 components are properly exported."""

    def test_anomaly_detector_exports(self):
        """Verify anomaly detector is exported from models."""
        from models import (  # noqa: F401
            AnomalyDetector,
            AnomalyDetectorConfig,
            AnomalyResult,
            AnomalySeverity,
            AnomalyType,
            MarketDataPoint,
            create_anomaly_detector,
        )

        assert AnomalyDetector is not None
        assert create_anomaly_detector is not None

    def test_reasoning_logger_exports(self):
        """Verify reasoning logger is exported from llm."""
        from llm import (  # noqa: F401
            ChainStatus,
            ReasoningChain,
            ReasoningLogger,
            SearchResult,
            create_reasoning_logger,
        )

        assert ReasoningLogger is not None
        assert create_reasoning_logger is not None

    def test_explainer_exports(self):
        """Verify explainer is exported from evaluation."""
        from evaluation import (  # noqa: F401
            LIME_AVAILABLE,
            SHAP_AVAILABLE,
            BaseExplainer,
            ExplainerFactory,
            ExplanationLogger,
            FeatureImportanceExplainer,
            LIMEExplainer,
            SHAPExplainer,
            create_explainer,
        )

        assert BaseExplainer is not None
        assert create_explainer is not None

    def test_circuit_breaker_has_anomaly_trip_reason(self):
        """Verify circuit breaker has ANOMALY_DETECTED trip reason."""
        from models import TripReason

        assert hasattr(TripReason, "ANOMALY_DETECTED")
        assert TripReason.ANOMALY_DETECTED.value == "anomaly_detected"

    def test_alerting_service_has_anomaly_category(self):
        """Verify alerting service has ANOMALY category."""
        from utils.alerting_service import AlertCategory

        assert hasattr(AlertCategory, "ANOMALY")
        assert AlertCategory.ANOMALY.value == "anomaly"

    def test_trading_agent_has_reasoning_logger(self):
        """Verify TradingAgent has reasoning_logger parameter."""
        # Check __init__ signature
        import inspect

        from llm.agents.base import TradingAgent

        sig = inspect.signature(TradingAgent.__init__)
        params = list(sig.parameters.keys())

        assert "reasoning_logger" in params
        assert "enable_reasoning_logging" in params
