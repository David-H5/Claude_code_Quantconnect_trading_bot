"""
Tests for Chain-of-Thought Reasoning Logger (UPGRADE-010 Sprint 1)

Tests the reasoning chain creation, tracking, search, and audit trail export.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from llm.decision_logger import ReasoningStep
from llm.reasoning_logger import (
    ChainStatus,
    ReasoningChain,
    ReasoningLogger,
    SearchResult,
    create_reasoning_logger,
)


class TestReasoningStep:
    """Tests for ReasoningStep integration."""

    @pytest.mark.unit
    def test_reasoning_step_creation(self):
        """Test creating a reasoning step."""
        step = ReasoningStep(
            step_number=1,
            thought="RSI indicates oversold",
            evidence="RSI = 28",
            confidence=0.85,
        )

        assert step.step_number == 1
        assert step.thought == "RSI indicates oversold"
        assert step.evidence == "RSI = 28"
        assert step.confidence == 0.85

    @pytest.mark.unit
    def test_reasoning_step_defaults(self):
        """Test reasoning step default values."""
        step = ReasoningStep(
            step_number=1,
            thought="Test thought",
        )

        assert step.evidence is None
        assert step.confidence == 0.5
        assert step.metadata == {}


class TestReasoningChain:
    """Tests for ReasoningChain class."""

    @pytest.fixture
    def sample_chain(self) -> ReasoningChain:
        """Create a sample reasoning chain."""
        return ReasoningChain(
            chain_id="test123",
            agent_name="technical_analyst",
            task="Analyze SPY for entry",
            started_at=datetime.utcnow(),
        )

    @pytest.mark.unit
    def test_chain_creation(self, sample_chain):
        """Test creating a reasoning chain."""
        assert sample_chain.chain_id == "test123"
        assert sample_chain.agent_name == "technical_analyst"
        assert sample_chain.task == "Analyze SPY for entry"
        assert sample_chain.status == ChainStatus.IN_PROGRESS
        assert len(sample_chain.steps) == 0

    @pytest.mark.unit
    def test_add_step(self, sample_chain):
        """Test adding steps to a chain."""
        step1 = sample_chain.add_step(
            thought="RSI at 35, oversold",
            evidence="RSI(14) = 35",
            confidence=0.8,
        )

        assert len(sample_chain.steps) == 1
        assert step1.step_number == 1
        assert step1.thought == "RSI at 35, oversold"

        step2 = sample_chain.add_step(
            thought="Price above SMA",
            confidence=0.9,
        )

        assert len(sample_chain.steps) == 2
        assert step2.step_number == 2

    @pytest.mark.unit
    def test_complete_chain(self, sample_chain):
        """Test completing a chain."""
        sample_chain.add_step("Step 1", confidence=0.8)
        sample_chain.complete("BUY", confidence=0.85)

        assert sample_chain.status == ChainStatus.COMPLETED
        assert sample_chain.final_decision == "BUY"
        assert sample_chain.final_confidence == 0.85
        assert sample_chain.completed_at is not None

    @pytest.mark.unit
    def test_fail_chain(self, sample_chain):
        """Test failing a chain."""
        sample_chain.fail("Insufficient data")

        assert sample_chain.status == ChainStatus.FAILED
        assert sample_chain.metadata["failure_reason"] == "Insufficient data"
        assert sample_chain.completed_at is not None

    @pytest.mark.unit
    def test_abandon_chain(self, sample_chain):
        """Test abandoning a chain."""
        sample_chain.abandon("User cancelled")

        assert sample_chain.status == ChainStatus.ABANDONED
        assert sample_chain.metadata["abandon_reason"] == "User cancelled"

    @pytest.mark.unit
    def test_average_confidence(self, sample_chain):
        """Test average confidence calculation."""
        sample_chain.add_step("Step 1", confidence=0.8)
        sample_chain.add_step("Step 2", confidence=0.9)
        sample_chain.add_step("Step 3", confidence=0.7)

        assert sample_chain.average_confidence == pytest.approx(0.8, rel=0.01)

    @pytest.mark.unit
    def test_average_confidence_empty(self, sample_chain):
        """Test average confidence with no steps."""
        assert sample_chain.average_confidence == 0.0

    @pytest.mark.unit
    def test_duration_ms(self, sample_chain):
        """Test duration calculation."""
        # Not completed yet
        assert sample_chain.duration_ms == 0.0

        # Complete the chain
        sample_chain.complete("BUY", 0.9)
        assert sample_chain.duration_ms >= 0.0

    @pytest.mark.unit
    def test_to_dict(self, sample_chain):
        """Test serialization to dictionary."""
        sample_chain.add_step("Test step", confidence=0.8)
        sample_chain.complete("HOLD", confidence=0.7)

        data = sample_chain.to_dict()

        assert data["chain_id"] == "test123"
        assert data["agent_name"] == "technical_analyst"
        assert data["status"] == "completed"
        assert len(data["steps"]) == 1
        assert data["final_decision"] == "HOLD"
        assert data["final_confidence"] == 0.7

    @pytest.mark.unit
    def test_from_dict(self, sample_chain):
        """Test deserialization from dictionary."""
        sample_chain.add_step("Test step", evidence="Evidence", confidence=0.8)
        sample_chain.complete("BUY", confidence=0.9)

        data = sample_chain.to_dict()
        restored = ReasoningChain.from_dict(data)

        assert restored.chain_id == sample_chain.chain_id
        assert restored.agent_name == sample_chain.agent_name
        assert restored.status == ChainStatus.COMPLETED
        assert len(restored.steps) == 1
        assert restored.final_decision == "BUY"


class TestReasoningLogger:
    """Tests for ReasoningLogger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def logger(self, temp_dir) -> ReasoningLogger:
        """Create a reasoning logger for testing."""
        return ReasoningLogger(
            storage_dir=temp_dir,
            auto_persist=True,
        )

    @pytest.mark.unit
    def test_start_chain(self, logger):
        """Test starting a new chain."""
        chain = logger.start_chain(
            agent_name="technical_analyst",
            task="Analyze AAPL",
        )

        assert chain.agent_name == "technical_analyst"
        assert chain.task == "Analyze AAPL"
        assert chain.status == ChainStatus.IN_PROGRESS
        assert chain.chain_id in logger._active_chains

    @pytest.mark.unit
    def test_complete_chain(self, logger):
        """Test completing a chain."""
        chain = logger.start_chain("analyst", "Test task")
        chain.add_step("Analysis step", confidence=0.8)

        completed = logger.complete_chain(
            chain.chain_id,
            decision="BUY",
            confidence=0.85,
        )

        assert completed is not None
        assert completed.status == ChainStatus.COMPLETED
        assert completed.final_decision == "BUY"
        assert chain.chain_id not in logger._active_chains
        assert completed in logger._completed_chains

    @pytest.mark.unit
    def test_complete_nonexistent_chain(self, logger):
        """Test completing a non-existent chain."""
        result = logger.complete_chain("nonexistent", "BUY", 0.9)
        assert result is None

    @pytest.mark.unit
    def test_get_chain_active(self, logger):
        """Test getting an active chain."""
        chain = logger.start_chain("analyst", "Task")
        retrieved = logger.get_chain(chain.chain_id)

        assert retrieved is not None
        assert retrieved.chain_id == chain.chain_id

    @pytest.mark.unit
    def test_get_chain_completed(self, logger):
        """Test getting a completed chain."""
        chain = logger.start_chain("analyst", "Task")
        logger.complete_chain(chain.chain_id, "HOLD", 0.7)

        retrieved = logger.get_chain(chain.chain_id)
        assert retrieved is not None
        assert retrieved.status == ChainStatus.COMPLETED

    @pytest.mark.unit
    def test_get_chains_by_agent(self, logger):
        """Test getting chains by agent name."""
        logger.start_chain("analyst1", "Task 1")
        logger.start_chain("analyst1", "Task 2")
        logger.start_chain("analyst2", "Task 3")

        chains = logger.get_chains_by_agent("analyst1")
        assert len(chains) == 2

    @pytest.mark.unit
    def test_search_reasoning_by_thought(self, logger):
        """Test searching by thought content."""
        chain = logger.start_chain("analyst", "Analyze SPY")
        chain.add_step("RSI oversold at 28", confidence=0.8)
        chain.add_step("MACD bullish crossover", confidence=0.7)
        logger.complete_chain(chain.chain_id, "BUY", 0.85)

        results = logger.search_reasoning("oversold")

        assert len(results) == 1
        assert results[0].chain_id == chain.chain_id
        assert len(results[0].matching_steps) == 1

    @pytest.mark.unit
    def test_search_reasoning_by_task(self, logger):
        """Test searching by task content."""
        chain = logger.start_chain("analyst", "Analyze volatility patterns")
        chain.add_step("VIX elevated", confidence=0.8)
        logger.complete_chain(chain.chain_id, "HOLD", 0.7)

        results = logger.search_reasoning("volatility")

        assert len(results) == 1
        assert results[0].task == "Analyze volatility patterns"

    @pytest.mark.unit
    def test_search_reasoning_with_agent_filter(self, logger):
        """Test search with agent filter."""
        chain1 = logger.start_chain("analyst1", "Task")
        chain1.add_step("Found signal", confidence=0.8)
        logger.complete_chain(chain1.chain_id, "BUY", 0.8)

        chain2 = logger.start_chain("analyst2", "Task")
        chain2.add_step("Found signal too", confidence=0.7)
        logger.complete_chain(chain2.chain_id, "SELL", 0.7)

        results = logger.search_reasoning("signal", agent_name="analyst1")

        assert len(results) == 1
        assert results[0].agent_name == "analyst1"

    @pytest.mark.unit
    def test_search_reasoning_limit(self, logger):
        """Test search result limit."""
        for i in range(5):
            chain = logger.start_chain("analyst", f"Task {i}")
            chain.add_step("Common pattern found", confidence=0.8)
            logger.complete_chain(chain.chain_id, "HOLD", 0.7)

        results = logger.search_reasoning("pattern", limit=3)
        assert len(results) == 3

    @pytest.mark.unit
    def test_export_audit_trail(self, logger, temp_dir):
        """Test audit trail export."""
        chain = logger.start_chain("analyst", "Analyze SPY")
        chain.add_step("Step 1", confidence=0.8)
        chain.add_step("Step 2", confidence=0.9)
        logger.complete_chain(chain.chain_id, "BUY", 0.85)

        export_path = os.path.join(temp_dir, "audit.json")
        count = logger.export_audit_trail(export_path)

        assert count == 1
        assert os.path.exists(export_path)

        with open(export_path) as f:
            data = json.load(f)

        assert data["total_chains"] == 1
        assert len(data["entries"]) == 1
        assert data["entries"][0]["step_count"] == 2
        assert "reasoning_steps" in data["entries"][0]

    @pytest.mark.unit
    def test_export_audit_trail_without_steps(self, logger, temp_dir):
        """Test audit trail export without steps."""
        chain = logger.start_chain("analyst", "Task")
        chain.add_step("Secret step", confidence=0.9)
        logger.complete_chain(chain.chain_id, "BUY", 0.9)

        export_path = os.path.join(temp_dir, "audit_no_steps.json")
        logger.export_audit_trail(export_path, include_steps=False)

        with open(export_path) as f:
            data = json.load(f)

        assert "reasoning_steps" not in data["entries"][0]

    @pytest.mark.unit
    def test_export_audit_trail_time_filter(self, logger, temp_dir):
        """Test audit trail with time filter."""
        chain = logger.start_chain("analyst", "Task")
        logger.complete_chain(chain.chain_id, "BUY", 0.9)

        # Filter to exclude the chain
        future_start = datetime.utcnow() + timedelta(hours=1)
        export_path = os.path.join(temp_dir, "audit_filtered.json")
        count = logger.export_audit_trail(export_path, start_time=future_start)

        assert count == 0

    @pytest.mark.unit
    def test_get_statistics_empty(self, logger):
        """Test statistics with no chains."""
        stats = logger.get_statistics()

        assert stats["total_chains"] == 0
        assert stats["active_chains"] == 0
        assert stats["completed_chains"] == 0

    @pytest.mark.unit
    def test_get_statistics(self, logger):
        """Test statistics calculation."""
        # Create some chains
        chain1 = logger.start_chain("analyst1", "Task 1")
        chain1.add_step("Step", confidence=0.8)
        chain1.add_step("Step", confidence=0.9)
        logger.complete_chain(chain1.chain_id, "BUY", 0.85)

        chain2 = logger.start_chain("analyst2", "Task 2")
        chain2.add_step("Step", confidence=0.7)
        logger.complete_chain(chain2.chain_id, "SELL", 0.7)

        logger.start_chain("analyst1", "Task 3")  # Active chain (unused var)

        stats = logger.get_statistics()

        assert stats["total_chains"] == 3
        assert stats["active_chains"] == 1
        assert stats["completed_chains"] == 2
        assert stats["total_steps"] == 3
        assert stats["average_steps_per_chain"] == 1.0
        assert stats["chains_by_agent"]["analyst1"] == 2
        assert stats["chains_by_agent"]["analyst2"] == 1

    @pytest.mark.unit
    def test_persistence(self, temp_dir):
        """Test chain persistence to disk."""
        logger = ReasoningLogger(storage_dir=temp_dir, auto_persist=True)

        chain = logger.start_chain("analyst", "Persistent task")
        chain.add_step("Important step", confidence=0.9)
        logger.complete_chain(chain.chain_id, "BUY", 0.95)

        # Check file exists
        expected_file = os.path.join(temp_dir, f"{chain.chain_id}.json")
        assert os.path.exists(expected_file)

        # Verify content
        with open(expected_file) as f:
            data = json.load(f)

        assert data["agent_name"] == "analyst"
        assert data["final_decision"] == "BUY"


class TestCreateReasoningLogger:
    """Tests for factory function."""

    @pytest.mark.unit
    def test_create_with_defaults(self):
        """Test factory with default settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = create_reasoning_logger(storage_dir=tmpdir)

            assert logger is not None
            assert logger.auto_persist is True

    @pytest.mark.unit
    def test_create_with_custom_settings(self):
        """Test factory with custom settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = create_reasoning_logger(
                storage_dir=tmpdir,
                auto_persist=False,
            )

            assert logger.auto_persist is False


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    @pytest.mark.unit
    def test_search_result_creation(self):
        """Test creating a search result."""
        step = ReasoningStep(1, "Test", confidence=0.8)
        result = SearchResult(
            chain_id="test123",
            agent_name="analyst",
            task="Task",
            matching_steps=[step],
            relevance_score=0.9,
            timestamp=datetime.utcnow(),
        )

        assert result.chain_id == "test123"
        assert len(result.matching_steps) == 1
        assert result.relevance_score == 0.9
