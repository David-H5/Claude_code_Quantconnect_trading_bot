"""
Tests for Agent Decision Logger

Tests verify the decision logger correctly:
- Logs agent decisions with full context
- Tracks reasoning chains and alternatives
- Persists to storage backends
- Analyzes decision patterns
"""

from datetime import datetime

import pytest

from llm.decision_logger import (
    AgentDecisionLog,
    Alternative,
    DecisionLogger,
    DecisionOutcome,
    DecisionType,
    InMemoryStorage,
    ReasoningStep,
    RiskAssessment,
    RiskLevel,
    create_decision_logger,
    generate_decision_report,
)


@pytest.fixture
def logger():
    """Create a decision logger with in-memory storage."""
    return DecisionLogger(storage=InMemoryStorage())


@pytest.fixture
def sample_reasoning_chain():
    """Create sample reasoning chain."""
    return [
        ReasoningStep(
            step_number=1,
            thought="Analyzing price action",
            evidence="Price above 200 SMA",
            confidence=0.7,
        ),
        ReasoningStep(
            step_number=2,
            thought="Checking volume",
            evidence="Volume 50% above average",
            confidence=0.8,
        ),
        ReasoningStep(
            step_number=3,
            thought="Confirming with RSI",
            evidence="RSI at 55, neutral",
            confidence=0.6,
        ),
    ]


@pytest.fixture
def sample_risk_assessment():
    """Create sample risk assessment."""
    return RiskAssessment(
        overall_level=RiskLevel.MEDIUM,
        factors=["market volatility", "position size"],
        mitigation_steps=["use stop loss", "scale in"],
        worst_case_scenario="5% loss if stopped out",
        probability_of_loss=0.3,
    )


class TestDecisionLogCreation:
    """Tests for decision log creation."""

    def test_log_decision_basic(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Can log a basic decision."""
        log = logger.log_decision(
            agent_name="technical_analyst",
            agent_role="analyst",
            decision_type=DecisionType.ANALYSIS,
            decision="BUY signal detected",
            confidence=0.85,
            context={"symbol": "SPY", "price": 450.0},
            query="Analyze SPY for entry",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        assert log is not None
        assert log.agent_name == "technical_analyst"
        assert log.decision == "BUY signal detected"
        assert log.confidence == 0.85
        assert len(log.reasoning_chain) == 3

    def test_log_generates_unique_id(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Each log gets a unique ID."""
        log1 = logger.log_decision(
            agent_name="agent1",
            agent_role="analyst",
            decision_type=DecisionType.ANALYSIS,
            decision="Decision 1",
            confidence=0.5,
            context={},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        log2 = logger.log_decision(
            agent_name="agent1",
            agent_role="analyst",
            decision_type=DecisionType.ANALYSIS,
            decision="Decision 2",
            confidence=0.5,
            context={},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        assert log1.log_id != log2.log_id

    def test_log_includes_timestamp(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Log includes timestamp."""
        before = datetime.utcnow()

        log = logger.log_decision(
            agent_name="agent",
            agent_role="analyst",
            decision_type=DecisionType.TRADE,
            decision="Execute trade",
            confidence=0.9,
            context={},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        after = datetime.utcnow()

        assert before <= log.timestamp <= after


class TestDecisionLogSerialization:
    """Tests for log serialization."""

    def test_to_dict(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Log can be converted to dictionary."""
        log = logger.log_decision(
            agent_name="agent",
            agent_role="analyst",
            decision_type=DecisionType.TRADE,
            decision="Execute trade",
            confidence=0.9,
            context={"symbol": "AAPL"},
            query="Should we trade?",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        log_dict = log.to_dict()

        assert log_dict["agent_name"] == "agent"
        assert log_dict["decision"] == "Execute trade"
        assert log_dict["decision_type"] == "trade"
        assert "reasoning_chain" in log_dict
        assert len(log_dict["reasoning_chain"]) == 3

    def test_from_dict(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Log can be restored from dictionary."""
        original = logger.log_decision(
            agent_name="agent",
            agent_role="analyst",
            decision_type=DecisionType.TRADE,
            decision="Execute trade",
            confidence=0.9,
            context={"symbol": "AAPL"},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        log_dict = original.to_dict()
        restored = AgentDecisionLog.from_dict(log_dict)

        assert restored.log_id == original.log_id
        assert restored.agent_name == original.agent_name
        assert restored.decision == original.decision
        assert restored.confidence == original.confidence


class TestDecisionLogQuery:
    """Tests for querying decision logs."""

    def test_get_by_agent(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Can get decisions by agent name."""
        # Log decisions from multiple agents
        for i in range(3):
            logger.log_decision(
                agent_name="agent_a",
                agent_role="analyst",
                decision_type=DecisionType.ANALYSIS,
                decision=f"Decision A{i}",
                confidence=0.8,
                context={},
                query="query",
                reasoning_chain=sample_reasoning_chain,
                risk_assessment=sample_risk_assessment,
            )

        for i in range(2):
            logger.log_decision(
                agent_name="agent_b",
                agent_role="trader",
                decision_type=DecisionType.TRADE,
                decision=f"Decision B{i}",
                confidence=0.7,
                context={},
                query="query",
                reasoning_chain=sample_reasoning_chain,
                risk_assessment=sample_risk_assessment,
            )

        agent_a_decisions = logger.get_decisions_by_agent("agent_a")
        agent_b_decisions = logger.get_decisions_by_agent("agent_b")

        assert len(agent_a_decisions) == 3
        assert len(agent_b_decisions) == 2

    def test_get_by_type(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Can get decisions by type."""
        logger.log_decision(
            agent_name="agent",
            agent_role="analyst",
            decision_type=DecisionType.ANALYSIS,
            decision="Analysis",
            confidence=0.8,
            context={},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        logger.log_decision(
            agent_name="agent",
            agent_role="trader",
            decision_type=DecisionType.TRADE,
            decision="Trade",
            confidence=0.9,
            context={},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        trades = logger.get_decisions_by_type(DecisionType.TRADE)
        analyses = logger.get_decisions_by_type(DecisionType.ANALYSIS)

        assert len(trades) == 1
        assert len(analyses) == 1

    def test_get_by_context(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Can get decisions by context value."""
        logger.log_decision(
            agent_name="agent",
            agent_role="analyst",
            decision_type=DecisionType.ANALYSIS,
            decision="SPY analysis",
            confidence=0.8,
            context={"symbol": "SPY"},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        logger.log_decision(
            agent_name="agent",
            agent_role="analyst",
            decision_type=DecisionType.ANALYSIS,
            decision="AAPL analysis",
            confidence=0.8,
            context={"symbol": "AAPL"},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        spy_decisions = logger.get_decisions_by_context("symbol", "SPY")

        assert len(spy_decisions) == 1
        assert spy_decisions[0].decision == "SPY analysis"

    def test_get_high_risk_decisions(self, logger, sample_reasoning_chain):
        """Can filter high risk decisions."""
        low_risk = RiskAssessment(
            overall_level=RiskLevel.LOW,
            factors=[],
            mitigation_steps=[],
            worst_case_scenario="minimal loss",
        )

        high_risk = RiskAssessment(
            overall_level=RiskLevel.HIGH,
            factors=["volatility"],
            mitigation_steps=["stop loss"],
            worst_case_scenario="significant loss",
        )

        logger.log_decision(
            agent_name="agent",
            agent_role="analyst",
            decision_type=DecisionType.TRADE,
            decision="Low risk trade",
            confidence=0.8,
            context={},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=low_risk,
        )

        logger.log_decision(
            agent_name="agent",
            agent_role="analyst",
            decision_type=DecisionType.TRADE,
            decision="High risk trade",
            confidence=0.8,
            context={},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=high_risk,
        )

        high_risk_logs = logger.get_high_risk_decisions()

        assert len(high_risk_logs) == 1
        assert high_risk_logs[0].decision == "High risk trade"

    def test_get_low_confidence_decisions(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Can filter low confidence decisions."""
        logger.log_decision(
            agent_name="agent",
            agent_role="analyst",
            decision_type=DecisionType.ANALYSIS,
            decision="High confidence",
            confidence=0.9,
            context={},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        logger.log_decision(
            agent_name="agent",
            agent_role="analyst",
            decision_type=DecisionType.ANALYSIS,
            decision="Low confidence",
            confidence=0.3,
            context={},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        low_conf = logger.get_low_confidence_decisions(threshold=0.5)

        assert len(low_conf) == 1
        assert low_conf[0].decision == "Low confidence"


class TestOutcomeTracking:
    """Tests for outcome tracking."""

    def test_update_outcome(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Can update decision outcome."""
        log = logger.log_decision(
            agent_name="agent",
            agent_role="trader",
            decision_type=DecisionType.TRADE,
            decision="Execute trade",
            confidence=0.9,
            context={},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        assert log.outcome == DecisionOutcome.PENDING

        success = logger.update_outcome(log.log_id, DecisionOutcome.EXECUTED)

        assert success
        assert log.outcome == DecisionOutcome.EXECUTED

    def test_update_outcome_with_metadata(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Can update outcome with additional metadata."""
        log = logger.log_decision(
            agent_name="agent",
            agent_role="trader",
            decision_type=DecisionType.TRADE,
            decision="Execute trade",
            confidence=0.9,
            context={},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        logger.update_outcome(
            log.log_id,
            DecisionOutcome.EXECUTED,
            metadata_update={"fill_price": 450.50, "slippage_bps": 2.5},
        )

        assert log.metadata["fill_price"] == 450.50
        assert log.metadata["slippage_bps"] == 2.5


class TestPatternAnalysis:
    """Tests for decision pattern analysis."""

    def test_analyze_empty_logs(self, logger):
        """Analysis handles empty logs."""
        analysis = logger.analyze_patterns()

        assert analysis.total_decisions == 0
        assert analysis.average_confidence == 0.0

    def test_analyze_patterns(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Analyzes patterns in logged decisions."""
        # Log various decisions
        for i in range(5):
            logger.log_decision(
                agent_name="analyst",
                agent_role="analyst",
                decision_type=DecisionType.ANALYSIS,
                decision=f"Analysis {i}",
                confidence=0.7 + i * 0.05,
                context={},
                query="query",
                reasoning_chain=sample_reasoning_chain,
                risk_assessment=sample_risk_assessment,
                execution_time_ms=100 + i * 10,
            )

        for i in range(3):
            logger.log_decision(
                agent_name="trader",
                agent_role="trader",
                decision_type=DecisionType.TRADE,
                decision=f"Trade {i}",
                confidence=0.8,
                context={},
                query="query",
                reasoning_chain=sample_reasoning_chain,
                risk_assessment=sample_risk_assessment,
                execution_time_ms=50,
            )

        analysis = logger.analyze_patterns()

        assert analysis.total_decisions == 8
        assert analysis.decisions_by_type["analysis"] == 5
        assert analysis.decisions_by_type["trade"] == 3
        assert analysis.decisions_by_agent["analyst"] == 5
        assert analysis.decisions_by_agent["trader"] == 3


class TestStorage:
    """Tests for storage backends."""

    def test_in_memory_storage_save_load(self, sample_reasoning_chain, sample_risk_assessment):
        """In-memory storage saves and loads logs."""
        storage = InMemoryStorage()
        logger = DecisionLogger(storage=storage)

        log = logger.log_decision(
            agent_name="agent",
            agent_role="analyst",
            decision_type=DecisionType.ANALYSIS,
            decision="Test",
            confidence=0.8,
            context={},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        # Flush to persist the log to storage (batch size is 100 by default)
        logger.flush()

        loaded = storage.load(log.log_id)

        assert loaded is not None
        assert loaded.log_id == log.log_id

    def test_in_memory_storage_max_size(self):
        """In-memory storage respects max size."""
        storage = InMemoryStorage(max_size=2)

        for i in range(5):
            log = AgentDecisionLog(
                log_id=f"log_{i}",
                timestamp=datetime.utcnow(),
                agent_name="agent",
                agent_role="analyst",
                decision_type=DecisionType.ANALYSIS,
                decision=f"Decision {i}",
                confidence=0.8,
                context={},
                query="query",
                reasoning_chain=[],
                final_reasoning="",
                alternatives_considered=[],
                risk_assessment=RiskAssessment(
                    overall_level=RiskLevel.LOW,
                    factors=[],
                    mitigation_steps=[],
                    worst_case_scenario="",
                ),
                execution_time_ms=100,
            )
            storage.save(log)

        assert len(storage.logs) == 2


class TestFactoryAndReport:
    """Tests for factory function and report generation."""

    def test_create_decision_logger_memory(self):
        """Factory creates memory logger."""
        logger = create_decision_logger(storage_type="memory")
        assert isinstance(logger.storage, InMemoryStorage)

    def test_generate_report(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Generates readable report."""
        for i in range(3):
            logger.log_decision(
                agent_name="analyst",
                agent_role="analyst",
                decision_type=DecisionType.ANALYSIS,
                decision=f"Analysis {i}",
                confidence=0.8,
                context={},
                query="query",
                reasoning_chain=sample_reasoning_chain,
                risk_assessment=sample_risk_assessment,
            )

        report = generate_decision_report(logger)

        assert "AGENT DECISION LOG REPORT" in report
        assert "Total Decisions: 3" in report
        assert "DECISIONS BY TYPE" in report
        assert "DECISIONS BY AGENT" in report


class TestAlternatives:
    """Tests for alternative tracking."""

    def test_log_with_alternatives(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Can log decisions with alternatives considered."""
        alternatives = [
            Alternative(
                description="Wait for pullback",
                reason_rejected="Momentum too strong",
                risk_level=RiskLevel.LOW,
            ),
            Alternative(
                description="Scale in gradually",
                reason_rejected="Position size already optimal",
                risk_level=RiskLevel.MEDIUM,
            ),
        ]

        log = logger.log_decision(
            agent_name="trader",
            agent_role="trader",
            decision_type=DecisionType.TRADE,
            decision="Enter full position now",
            confidence=0.85,
            context={},
            query="How to enter trade?",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
            alternatives=alternatives,
        )

        assert len(log.alternatives_considered) == 2
        assert log.alternatives_considered[0].description == "Wait for pullback"


class TestClearAndFlush:
    """Tests for clear and flush operations."""

    def test_clear_logs(self, logger, sample_reasoning_chain, sample_risk_assessment):
        """Can clear all logs."""
        logger.log_decision(
            agent_name="agent",
            agent_role="analyst",
            decision_type=DecisionType.ANALYSIS,
            decision="Test",
            confidence=0.8,
            context={},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        assert len(logger.logs) == 1

        logger.clear()

        assert len(logger.logs) == 0

    def test_flush_pending(self, sample_reasoning_chain, sample_risk_assessment):
        """Can flush pending logs to storage."""
        storage = InMemoryStorage()
        logger = DecisionLogger(storage=storage, batch_size=100)

        # Log without triggering auto-flush
        log = logger.log_decision(
            agent_name="agent",
            agent_role="analyst",
            decision_type=DecisionType.ANALYSIS,
            decision="Test",
            confidence=0.8,
            context={},
            query="query",
            reasoning_chain=sample_reasoning_chain,
            risk_assessment=sample_risk_assessment,
        )

        # Manual flush
        logger.flush()

        # Should be in storage
        loaded = storage.load(log.log_id)
        assert loaded is not None
