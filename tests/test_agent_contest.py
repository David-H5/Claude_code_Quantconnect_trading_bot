"""
Tests for Agent Performance Contest System (UPGRADE-010 Sprint 2)

Tests ELO rating system, confidence-weighted voting, and contest management.
"""

import tempfile
from datetime import datetime

import pytest

from evaluation.agent_contest import (
    AgentRanking,
    AgentRecord,
    ContestManager,
    ELOSystem,
    Prediction,
    PredictionOutcome,
    VoteResult,
    VotingEngine,
    create_contest_manager,
)


class TestPrediction:
    """Tests for Prediction dataclass."""

    @pytest.mark.unit
    def test_prediction_creation(self):
        """Test creating a prediction."""
        pred = Prediction(
            prediction_id="pred_001",
            agent_name="technical_analyst",
            prediction="BUY",
            confidence=0.85,
            symbol="SPY",
            timestamp=datetime.utcnow(),
        )

        assert pred.prediction_id == "pred_001"
        assert pred.agent_name == "technical_analyst"
        assert pred.prediction == "BUY"
        assert pred.confidence == 0.85
        assert pred.symbol == "SPY"
        assert pred.outcome == PredictionOutcome.PENDING

    @pytest.mark.unit
    def test_prediction_to_dict(self):
        """Test serialization."""
        pred = Prediction(
            prediction_id="pred_001",
            agent_name="analyst",
            prediction="SELL",
            confidence=0.75,
            symbol="AAPL",
            timestamp=datetime.utcnow(),
        )

        data = pred.to_dict()

        assert data["prediction_id"] == "pred_001"
        assert data["prediction"] == "SELL"
        assert data["outcome"] == "pending"

    @pytest.mark.unit
    def test_prediction_from_dict(self):
        """Test deserialization."""
        data = {
            "prediction_id": "pred_002",
            "agent_name": "analyst",
            "prediction": "HOLD",
            "confidence": 0.60,
            "symbol": "MSFT",
            "timestamp": "2025-12-02T10:00:00",
            "outcome": "correct",
            "actual": "HOLD",
        }

        pred = Prediction.from_dict(data)

        assert pred.prediction_id == "pred_002"
        assert pred.outcome == PredictionOutcome.CORRECT
        assert pred.actual == "HOLD"


class TestAgentRecord:
    """Tests for AgentRecord dataclass."""

    @pytest.mark.unit
    def test_agent_record_creation(self):
        """Test creating an agent record."""
        record = AgentRecord(agent_name="technical_analyst")

        assert record.agent_name == "technical_analyst"
        assert record.elo_rating == 1500.0
        assert record.wins == 0
        assert record.losses == 0
        assert record.draws == 0

    @pytest.mark.unit
    def test_total_predictions(self):
        """Test total predictions calculation."""
        record = AgentRecord(agent_name="analyst")
        record.wins = 10
        record.losses = 5
        record.draws = 2

        assert record.total_predictions == 17

    @pytest.mark.unit
    def test_win_rate(self):
        """Test win rate calculation."""
        record = AgentRecord(agent_name="analyst")
        record.wins = 7
        record.losses = 3
        record.draws = 0

        assert record.win_rate == 0.7

    @pytest.mark.unit
    def test_win_rate_empty(self):
        """Test win rate with no predictions."""
        record = AgentRecord(agent_name="analyst")
        assert record.win_rate == 0.0

    @pytest.mark.unit
    def test_accuracy(self):
        """Test accuracy with partial credit."""
        record = AgentRecord(agent_name="analyst")
        record.wins = 6
        record.losses = 2
        record.draws = 2  # Draws count as 0.5

        # Accuracy = (6 + 0.5*2) / 10 = 7 / 10 = 0.7
        assert record.accuracy == 0.7

    @pytest.mark.unit
    def test_average_confidence(self):
        """Test average confidence calculation."""
        record = AgentRecord(agent_name="analyst")
        record.predictions = [
            Prediction(
                prediction_id="p1",
                agent_name="analyst",
                prediction="BUY",
                confidence=0.8,
                symbol="SPY",
                timestamp=datetime.utcnow(),
            ),
            Prediction(
                prediction_id="p2",
                agent_name="analyst",
                prediction="SELL",
                confidence=0.6,
                symbol="SPY",
                timestamp=datetime.utcnow(),
            ),
        ]

        assert record.average_confidence == 0.7

    @pytest.mark.unit
    def test_calibration_error(self):
        """Test calibration error calculation."""
        record = AgentRecord(agent_name="analyst")
        record.wins = 8
        record.losses = 2
        record.draws = 0
        # Accuracy = 0.8

        record.predictions = [
            Prediction(
                prediction_id=f"p{i}",
                agent_name="analyst",
                prediction="BUY",
                confidence=0.9,  # Average confidence = 0.9
                symbol="SPY",
                timestamp=datetime.utcnow(),
            )
            for i in range(10)
        ]

        # Calibration error = |0.9 - 0.8| = 0.1
        assert record.calibration_error == pytest.approx(0.1, rel=0.01)

    @pytest.mark.unit
    def test_add_prediction_limits_history(self):
        """Test that prediction history is limited to 100."""
        record = AgentRecord(agent_name="analyst")

        for i in range(110):
            pred = Prediction(
                prediction_id=f"pred_{i}",
                agent_name="analyst",
                prediction="BUY",
                confidence=0.8,
                symbol="SPY",
                timestamp=datetime.utcnow(),
            )
            record.add_prediction(pred)

        assert len(record.predictions) == 100

    @pytest.mark.unit
    def test_record_outcome(self):
        """Test recording outcomes."""
        record = AgentRecord(agent_name="analyst")

        record.record_outcome(PredictionOutcome.CORRECT)
        assert record.wins == 1

        record.record_outcome(PredictionOutcome.INCORRECT)
        assert record.losses == 1

        record.record_outcome(PredictionOutcome.PARTIAL)
        assert record.draws == 1

    @pytest.mark.unit
    def test_to_dict_and_from_dict(self):
        """Test serialization roundtrip."""
        record = AgentRecord(
            agent_name="analyst",
            elo_rating=1600.0,
            wins=10,
            losses=5,
        )

        data = record.to_dict()
        restored = AgentRecord.from_dict(data)

        assert restored.agent_name == record.agent_name
        assert restored.elo_rating == record.elo_rating
        assert restored.wins == record.wins


class TestELOSystem:
    """Tests for ELO rating system."""

    @pytest.fixture
    def elo(self) -> ELOSystem:
        """Create ELO system for testing."""
        return ELOSystem(k_factor=32.0)

    @pytest.mark.unit
    def test_expected_score_equal_ratings(self, elo):
        """Test expected score with equal ratings."""
        expected = elo.expected_score(1500, 1500)
        assert expected == 0.5

    @pytest.mark.unit
    def test_expected_score_higher_rating(self, elo):
        """Test expected score when rating A is higher."""
        expected = elo.expected_score(1600, 1400)
        assert expected > 0.5
        assert expected < 1.0

    @pytest.mark.unit
    def test_expected_score_lower_rating(self, elo):
        """Test expected score when rating A is lower."""
        expected = elo.expected_score(1400, 1600)
        assert expected < 0.5
        assert expected > 0.0

    @pytest.mark.unit
    def test_update_rating_win(self, elo):
        """Test rating update on win."""
        new_rating = elo.update_rating(
            current_rating=1500,
            expected=0.5,
            actual=1.0,  # Win
        )

        # Should increase
        assert new_rating > 1500
        # Max increase is K-factor
        assert new_rating <= 1500 + 32

    @pytest.mark.unit
    def test_update_rating_loss(self, elo):
        """Test rating update on loss."""
        new_rating = elo.update_rating(
            current_rating=1500,
            expected=0.5,
            actual=0.0,  # Loss
        )

        # Should decrease
        assert new_rating < 1500
        # Max decrease is K-factor
        assert new_rating >= 1500 - 32

    @pytest.mark.unit
    def test_update_rating_draw(self, elo):
        """Test rating update on draw."""
        new_rating = elo.update_rating(
            current_rating=1500,
            expected=0.5,
            actual=0.5,  # Draw
        )

        # Should stay same (expected = actual)
        assert new_rating == 1500

    @pytest.mark.unit
    def test_update_rating_confidence_scaling(self, elo):
        """Test that confidence scales K-factor."""
        # High confidence win
        high_conf = elo.update_rating(1500, 0.5, 1.0, confidence=1.0)

        # Low confidence win
        low_conf = elo.update_rating(1500, 0.5, 1.0, confidence=0.5)

        # High confidence should change more
        assert high_conf - 1500 > low_conf - 1500

    @pytest.mark.unit
    def test_rating_bounds(self, elo):
        """Test rating stays within bounds."""
        # Try to go below minimum
        low = elo.update_rating(100, 0.99, 0.0)
        assert low >= elo.min_rating

        # Try to go above maximum
        high = elo.update_rating(3000, 0.01, 1.0)
        assert high <= elo.max_rating

    @pytest.mark.unit
    def test_update_from_prediction(self, elo):
        """Test updating from prediction outcome."""
        record = AgentRecord(agent_name="analyst", elo_rating=1500.0)

        new_rating = elo.update_from_prediction(
            record=record,
            outcome=PredictionOutcome.CORRECT,
            confidence=0.8,
        )

        assert new_rating > 1500
        assert record.elo_rating == new_rating


class TestVotingEngine:
    """Tests for voting engine."""

    @pytest.fixture
    def voting(self) -> VotingEngine:
        """Create voting engine for testing."""
        return VotingEngine(min_confidence=0.5, elo_weight_factor=0.3)

    @pytest.mark.unit
    def test_calculate_weight_base_elo(self, voting):
        """Test weight calculation at base ELO."""
        weight = voting.calculate_weight(
            confidence=0.8,
            elo_rating=1500.0,
        )

        # At base ELO, weight should equal confidence
        assert weight == pytest.approx(0.8, rel=0.01)

    @pytest.mark.unit
    def test_calculate_weight_high_elo(self, voting):
        """Test weight calculation with high ELO."""
        weight = voting.calculate_weight(
            confidence=0.8,
            elo_rating=1800.0,  # 300 above base
        )

        # Should be higher than base
        assert weight > 0.8

    @pytest.mark.unit
    def test_calculate_weight_low_elo(self, voting):
        """Test weight calculation with low ELO."""
        weight = voting.calculate_weight(
            confidence=0.8,
            elo_rating=1200.0,  # 300 below base
        )

        # Should be lower than base
        assert weight < 0.8

    @pytest.mark.unit
    def test_weighted_vote_unanimous(self, voting):
        """Test vote with unanimous agreement."""
        predictions = {
            "agent1": ("BUY", 0.8),
            "agent2": ("BUY", 0.7),
            "agent3": ("BUY", 0.9),
        }
        records = {
            "agent1": AgentRecord("agent1", elo_rating=1500),
            "agent2": AgentRecord("agent2", elo_rating=1500),
            "agent3": AgentRecord("agent3", elo_rating=1500),
        }

        result = voting.weighted_vote(predictions, records)

        assert result.consensus == "BUY"
        assert result.confidence == 1.0
        assert result.agreement_score == 1.0

    @pytest.mark.unit
    def test_weighted_vote_split(self, voting):
        """Test vote with split decision."""
        predictions = {
            "agent1": ("BUY", 0.8),
            "agent2": ("SELL", 0.9),
        }
        records = {
            "agent1": AgentRecord("agent1", elo_rating=1500),
            "agent2": AgentRecord("agent2", elo_rating=1500),
        }

        result = voting.weighted_vote(predictions, records)

        # SELL should win (higher confidence)
        assert result.consensus == "SELL"
        assert 0 < result.confidence < 1
        assert len(result.vote_breakdown) == 2

    @pytest.mark.unit
    def test_weighted_vote_elo_tiebreaker(self, voting):
        """Test that ELO breaks ties."""
        predictions = {
            "agent1": ("BUY", 0.8),
            "agent2": ("SELL", 0.8),  # Same confidence
        }
        records = {
            "agent1": AgentRecord("agent1", elo_rating=1800),  # Higher ELO
            "agent2": AgentRecord("agent2", elo_rating=1200),
        }

        result = voting.weighted_vote(predictions, records)

        # BUY should win (higher ELO)
        assert result.consensus == "BUY"

    @pytest.mark.unit
    def test_weighted_vote_filters_low_confidence(self, voting):
        """Test that low confidence predictions are filtered."""
        predictions = {
            "agent1": ("BUY", 0.3),  # Below threshold
            "agent2": ("SELL", 0.8),
        }
        records = {
            "agent1": AgentRecord("agent1", elo_rating=1500),
            "agent2": AgentRecord("agent2", elo_rating=1500),
        }

        result = voting.weighted_vote(predictions, records)

        # Only agent2 should participate
        assert result.consensus == "SELL"
        assert len(result.participating_agents) == 1

    @pytest.mark.unit
    def test_weighted_vote_empty(self, voting):
        """Test vote with no valid predictions."""
        predictions = {
            "agent1": ("BUY", 0.3),  # Below threshold
        }
        records = {
            "agent1": AgentRecord("agent1", elo_rating=1500),
        }

        result = voting.weighted_vote(predictions, records)

        assert result.consensus == "ABSTAIN"
        assert result.confidence == 0.0


class TestContestManager:
    """Tests for contest manager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def manager(self, temp_dir) -> ContestManager:
        """Create contest manager for testing."""
        return ContestManager(
            storage_dir=temp_dir,
            auto_persist=True,
        )

    @pytest.mark.unit
    def test_register_agent(self, manager):
        """Test agent registration."""
        record = manager.register_agent("technical_analyst")

        assert record.agent_name == "technical_analyst"
        assert record.elo_rating == 1500.0
        assert manager.get_agent("technical_analyst") is not None

    @pytest.mark.unit
    def test_register_agent_custom_rating(self, manager):
        """Test registration with custom rating."""
        record = manager.register_agent(
            "expert",
            initial_rating=1800.0,
        )

        assert record.elo_rating == 1800.0

    @pytest.mark.unit
    def test_register_agent_idempotent(self, manager):
        """Test that registering same agent returns existing."""
        record1 = manager.register_agent("analyst")
        record1.wins = 5

        record2 = manager.register_agent("analyst")

        assert record2.wins == 5
        assert record1 is record2

    @pytest.mark.unit
    def test_track_prediction(self, manager):
        """Test tracking a prediction."""
        pred = manager.track_prediction(
            agent_name="analyst",
            prediction="BUY",
            confidence=0.85,
            symbol="SPY",
        )

        assert pred.agent_name == "analyst"
        assert pred.prediction == "BUY"
        assert pred.symbol == "SPY"

    @pytest.mark.unit
    def test_track_prediction_auto_registers(self, manager):
        """Test that tracking auto-registers unknown agents."""
        manager.track_prediction(
            agent_name="new_agent",
            prediction="SELL",
            confidence=0.7,
            symbol="AAPL",
        )

        assert manager.get_agent("new_agent") is not None

    @pytest.mark.unit
    def test_record_outcome_correct(self, manager):
        """Test recording correct outcome."""
        manager.track_prediction("analyst", "BUY", 0.8, "SPY")

        results = manager.record_outcome("SPY", "BUY")

        assert len(results) == 1
        agent, outcome, rating = results[0]
        assert outcome == PredictionOutcome.CORRECT
        assert rating > 1500  # ELO increased

    @pytest.mark.unit
    def test_record_outcome_incorrect(self, manager):
        """Test recording incorrect outcome."""
        manager.track_prediction("analyst", "BUY", 0.8, "SPY")

        results = manager.record_outcome("SPY", "SELL")

        agent, outcome, rating = results[0]
        assert outcome == PredictionOutcome.INCORRECT
        assert rating < 1500  # ELO decreased

    @pytest.mark.unit
    def test_record_outcome_partial(self, manager):
        """Test recording partial match."""
        manager.track_prediction("analyst", "BUY", 0.8, "SPY")

        results = manager.record_outcome(
            "SPY",
            "STRONG_BUY",
            partial_matches=["BUY"],
        )

        agent, outcome, rating = results[0]
        assert outcome == PredictionOutcome.PARTIAL

    @pytest.mark.unit
    def test_get_rankings(self, manager):
        """Test getting agent rankings."""
        # Create agents with different ratings
        manager.register_agent("agent1", initial_rating=1600)
        manager.register_agent("agent2", initial_rating=1500)
        manager.register_agent("agent3", initial_rating=1700)

        rankings = manager.get_rankings()

        assert len(rankings) == 3
        assert rankings[0].agent_name == "agent3"
        assert rankings[0].rank == 1
        assert rankings[2].agent_name == "agent2"
        assert rankings[2].rank == 3

    @pytest.mark.unit
    def test_weighted_vote(self, manager):
        """Test weighted voting through manager."""
        manager.register_agent("agent1", initial_rating=1500)
        manager.register_agent("agent2", initial_rating=1500)

        predictions = {
            "agent1": ("BUY", 0.8),
            "agent2": ("SELL", 0.6),
        }

        result = manager.weighted_vote(predictions)

        assert result.consensus == "BUY"
        assert result.confidence > 0.5

    @pytest.mark.unit
    def test_compare_agents(self, manager):
        """Test agent comparison."""
        manager.register_agent("expert", initial_rating=1800)
        manager.register_agent("novice", initial_rating=1200)

        comparison = manager.compare_agents("expert", "novice")

        assert comparison["elo_difference"] == 600
        assert comparison["expected_win_rate_a"] > 0.5

    @pytest.mark.unit
    def test_compare_agents_not_found(self, manager):
        """Test comparison with unknown agent."""
        manager.register_agent("agent1")

        comparison = manager.compare_agents("agent1", "unknown")

        assert "error" in comparison

    @pytest.mark.unit
    def test_get_statistics(self, manager):
        """Test getting statistics."""
        manager.register_agent("agent1")
        manager.register_agent("agent2", initial_rating=1600)
        manager.track_prediction("agent1", "BUY", 0.8, "SPY")
        manager.record_outcome("SPY", "BUY")

        stats = manager.get_statistics()

        assert stats["total_agents"] == 2
        assert stats["total_predictions"] == 1
        assert stats["average_elo"] > 1500

    @pytest.mark.unit
    def test_statistics_empty(self, manager):
        """Test statistics with no agents."""
        stats = manager.get_statistics()

        assert stats["total_agents"] == 0
        assert stats["total_predictions"] == 0

    @pytest.mark.unit
    def test_persistence(self, temp_dir):
        """Test saving and loading contest data."""
        # Create and populate manager
        manager1 = ContestManager(storage_dir=temp_dir, auto_persist=True)
        manager1.register_agent("analyst", initial_rating=1600)
        manager1.track_prediction("analyst", "BUY", 0.8, "SPY")
        manager1.record_outcome("SPY", "BUY")

        # Create new manager and load
        manager2 = ContestManager(storage_dir=temp_dir, auto_persist=False)
        success = manager2.load()

        assert success
        agent = manager2.get_agent("analyst")
        assert agent is not None
        assert agent.wins == 1


class TestCreateContestManager:
    """Tests for factory function."""

    @pytest.mark.unit
    def test_create_with_defaults(self):
        """Test factory with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_contest_manager(storage_dir=tmpdir)

            assert manager is not None
            assert manager.elo_system.k_factor == 32.0

    @pytest.mark.unit
    def test_create_with_custom_settings(self):
        """Test factory with custom settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_contest_manager(
                k_factor=24.0,
                min_confidence=0.6,
                storage_dir=tmpdir,
            )

            assert manager.elo_system.k_factor == 24.0
            assert manager.voting_engine.min_confidence == 0.6


class TestAgentRanking:
    """Tests for AgentRanking dataclass."""

    @pytest.mark.unit
    def test_ranking_creation(self):
        """Test creating a ranking."""
        ranking = AgentRanking(
            rank=1,
            agent_name="expert",
            elo_rating=1800.0,
            win_rate=0.75,
            accuracy=0.80,
            total_predictions=100,
            calibration_error=0.05,
        )

        assert ranking.rank == 1
        assert ranking.agent_name == "expert"
        assert ranking.elo_rating == 1800.0


class TestVoteResult:
    """Tests for VoteResult dataclass."""

    @pytest.mark.unit
    def test_vote_result_creation(self):
        """Test creating a vote result."""
        result = VoteResult(
            consensus="BUY",
            confidence=0.85,
            vote_breakdown={"BUY": 0.7, "SELL": 0.3},
            participating_agents=["agent1", "agent2"],
            total_weight=1.5,
            agreement_score=0.7,
        )

        assert result.consensus == "BUY"
        assert result.confidence == 0.85
        assert len(result.vote_breakdown) == 2
