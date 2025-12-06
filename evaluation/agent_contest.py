"""
Agent Performance Contest System (UPGRADE-010 Sprint 2)

Implements ELO-based agent ranking and confidence-weighted voting
for multi-agent trading decision systems.

Features:
- ELO rating system for agent performance tracking
- Confidence-weighted voting for consensus decisions
- Historical performance storage and analysis
- Agent comparison and ranking

QuantConnect Compatible: Yes

Usage:
    from evaluation.agent_contest import (
        ContestManager,
        create_contest_manager,
    )

    # Create contest manager
    contest = create_contest_manager()

    # Register agents
    contest.register_agent("technical_analyst")
    contest.register_agent("sentiment_analyst")

    # Track predictions
    contest.track_prediction(
        agent_name="technical_analyst",
        prediction="BUY",
        confidence=0.85,
        symbol="SPY",
    )

    # Record actual outcome
    contest.record_outcome(symbol="SPY", actual="BUY")

    # Get rankings
    rankings = contest.get_rankings()

    # Weighted vote
    predictions = {
        "technical_analyst": ("BUY", 0.85),
        "sentiment_analyst": ("HOLD", 0.60),
    }
    consensus = contest.weighted_vote(predictions)
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class PredictionOutcome(Enum):
    """Outcome of a prediction."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"
    PENDING = "pending"


@dataclass
class Prediction:
    """A single prediction record."""

    prediction_id: str
    agent_name: str
    prediction: str
    confidence: float
    symbol: str
    timestamp: datetime
    outcome: PredictionOutcome = PredictionOutcome.PENDING
    actual: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction_id": self.prediction_id,
            "agent_name": self.agent_name,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "outcome": self.outcome.value,
            "actual": self.actual,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Prediction:
        """Create from dictionary."""
        return cls(
            prediction_id=data["prediction_id"],
            agent_name=data["agent_name"],
            prediction=data["prediction"],
            confidence=data["confidence"],
            symbol=data["symbol"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            outcome=PredictionOutcome(data["outcome"]),
            actual=data.get("actual"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ELOHistoryEntry:
    """Single entry in ELO history."""

    timestamp: datetime
    elo_rating: float
    change: float
    outcome: PredictionOutcome | None = None
    prediction_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "elo_rating": self.elo_rating,
            "change": self.change,
            "outcome": self.outcome.value if self.outcome else None,
            "prediction_id": self.prediction_id,
        }


@dataclass
class AgentRecord:
    """Performance record for a single agent."""

    agent_name: str
    elo_rating: float = 1500.0
    predictions: list[Prediction] = field(default_factory=list)
    elo_history: list[ELOHistoryEntry] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    draws: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_predictions(self) -> int:
        """Total number of predictions made."""
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        """Win rate as a percentage."""
        if self.total_predictions == 0:
            return 0.0
        return self.wins / self.total_predictions

    @property
    def accuracy(self) -> float:
        """Accuracy including partial credit for draws."""
        if self.total_predictions == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total_predictions

    @property
    def average_confidence(self) -> float:
        """Average confidence across all predictions."""
        if not self.predictions:
            return 0.0
        return sum(p.confidence for p in self.predictions) / len(self.predictions)

    @property
    def calibration_error(self) -> float:
        """Absolute difference between confidence and accuracy."""
        return abs(self.average_confidence - self.accuracy)

    def add_prediction(self, prediction: Prediction) -> None:
        """Add a prediction to history."""
        self.predictions.append(prediction)
        self.last_updated = datetime.utcnow()

        # Keep last 100 predictions
        if len(self.predictions) > 100:
            self.predictions = self.predictions[-100:]

    def record_outcome(self, outcome: PredictionOutcome) -> None:
        """Record an outcome."""
        if outcome == PredictionOutcome.CORRECT:
            self.wins += 1
        elif outcome == PredictionOutcome.INCORRECT:
            self.losses += 1
        elif outcome == PredictionOutcome.PARTIAL:
            self.draws += 1
        self.last_updated = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "elo_rating": self.elo_rating,
            "predictions": [p.to_dict() for p in self.predictions[-20:]],
            "elo_history": [e.to_dict() for e in self.elo_history[-100:]],
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentRecord:
        """Create from dictionary."""
        record = cls(
            agent_name=data["agent_name"],
            elo_rating=data["elo_rating"],
            wins=data["wins"],
            losses=data["losses"],
            draws=data["draws"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            metadata=data.get("metadata", {}),
        )
        record.predictions = [Prediction.from_dict(p) for p in data.get("predictions", [])]
        # Restore ELO history
        record.elo_history = []
        for entry_data in data.get("elo_history", []):
            outcome = None
            if entry_data.get("outcome"):
                outcome = PredictionOutcome(entry_data["outcome"])
            record.elo_history.append(
                ELOHistoryEntry(
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    elo_rating=entry_data["elo_rating"],
                    change=entry_data["change"],
                    outcome=outcome,
                    prediction_id=entry_data.get("prediction_id"),
                )
            )
        return record


@dataclass
class AgentRanking:
    """Ranking information for an agent."""

    rank: int
    agent_name: str
    elo_rating: float
    win_rate: float
    accuracy: float
    total_predictions: int
    calibration_error: float


class ELOSystem:
    """
    ELO rating system for agent performance.

    Uses the standard ELO formula with configurable K-factor.
    Agents gain/lose rating based on prediction accuracy.
    """

    def __init__(
        self,
        k_factor: float = 32.0,
        base_rating: float = 1500.0,
        min_rating: float = 100.0,
        max_rating: float = 3000.0,
    ):
        """
        Initialize ELO system.

        Args:
            k_factor: Maximum rating change per outcome (default 32)
            base_rating: Starting rating for new agents
            min_rating: Minimum possible rating
            max_rating: Maximum possible rating
        """
        self.k_factor = k_factor
        self.base_rating = base_rating
        self.min_rating = min_rating
        self.max_rating = max_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A vs player B.

        Uses standard ELO formula: E = 1 / (1 + 10^((Rb - Ra) / 400))

        Args:
            rating_a: Rating of player A
            rating_b: Rating of player B

        Returns:
            Expected score (0-1) for player A
        """
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def update_rating(
        self,
        current_rating: float,
        expected: float,
        actual: float,
        confidence: float = 1.0,
    ) -> float:
        """
        Update rating based on outcome.

        Args:
            current_rating: Current ELO rating
            expected: Expected score (0-1)
            actual: Actual score (1=win, 0.5=draw, 0=loss)
            confidence: Agent's stated confidence (scales K-factor)

        Returns:
            New ELO rating
        """
        # Scale K-factor by confidence
        effective_k = self.k_factor * confidence

        # Standard ELO update
        new_rating = current_rating + effective_k * (actual - expected)

        # Clamp to valid range
        return max(self.min_rating, min(self.max_rating, new_rating))

    def update_from_prediction(
        self,
        record: AgentRecord,
        outcome: PredictionOutcome,
        confidence: float,
        opponent_rating: float | None = None,
    ) -> float:
        """
        Update agent rating based on prediction outcome.

        Args:
            record: Agent's record
            outcome: Prediction outcome
            confidence: Agent's stated confidence
            opponent_rating: Rating of "opponent" (market/baseline), defaults to base

        Returns:
            New ELO rating
        """
        if opponent_rating is None:
            opponent_rating = self.base_rating

        # Calculate expected score
        expected = self.expected_score(record.elo_rating, opponent_rating)

        # Map outcome to actual score
        if outcome == PredictionOutcome.CORRECT:
            actual = 1.0
        elif outcome == PredictionOutcome.PARTIAL:
            actual = 0.5
        else:
            actual = 0.0

        # Update and return new rating
        old_rating = record.elo_rating
        new_rating = self.update_rating(
            record.elo_rating,
            expected,
            actual,
            confidence,
        )
        record.elo_rating = new_rating

        # Track ELO history
        history_entry = ELOHistoryEntry(
            timestamp=datetime.utcnow(),
            elo_rating=new_rating,
            change=new_rating - old_rating,
            outcome=outcome,
        )
        record.elo_history.append(history_entry)

        # Keep last 500 history entries
        if len(record.elo_history) > 500:
            record.elo_history = record.elo_history[-500:]

        return new_rating


@dataclass
class VoteResult:
    """Result of a weighted vote."""

    consensus: str
    confidence: float
    vote_breakdown: dict[str, float]
    participating_agents: list[str]
    total_weight: float
    agreement_score: float


class VotingEngine:
    """
    Confidence-weighted voting engine.

    Combines agent predictions using ELO-weighted confidence voting.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        elo_weight_factor: float = 0.3,
    ):
        """
        Initialize voting engine.

        Args:
            min_confidence: Minimum confidence to participate in vote
            elo_weight_factor: How much ELO affects vote weight (0-1)
        """
        self.min_confidence = min_confidence
        self.elo_weight_factor = elo_weight_factor

    def calculate_weight(
        self,
        confidence: float,
        elo_rating: float,
        base_elo: float = 1500.0,
    ) -> float:
        """
        Calculate vote weight for an agent.

        Weight = confidence * (1 + elo_factor * (rating - base) / 1000)

        Args:
            confidence: Agent's stated confidence
            elo_rating: Agent's ELO rating
            base_elo: Baseline ELO rating

        Returns:
            Vote weight
        """
        # ELO modifier: above 1500 adds weight, below reduces
        elo_modifier = 1.0 + self.elo_weight_factor * (elo_rating - base_elo) / 1000.0

        # Clamp ELO modifier to reasonable range
        elo_modifier = max(0.5, min(2.0, elo_modifier))

        return confidence * elo_modifier

    def weighted_vote(
        self,
        predictions: dict[str, tuple[str, float]],
        agent_records: dict[str, AgentRecord],
    ) -> VoteResult:
        """
        Perform weighted vote across agents.

        Args:
            predictions: Dict of agent_name -> (prediction, confidence)
            agent_records: Dict of agent_name -> AgentRecord

        Returns:
            VoteResult with consensus and breakdown
        """
        vote_weights: dict[str, float] = {}
        participating_agents = []

        for agent_name, (prediction, confidence) in predictions.items():
            # Skip low confidence predictions
            if confidence < self.min_confidence:
                continue

            # Get ELO rating
            record = agent_records.get(agent_name)
            elo_rating = record.elo_rating if record else 1500.0

            # Calculate weight
            weight = self.calculate_weight(confidence, elo_rating)

            # Accumulate votes
            if prediction not in vote_weights:
                vote_weights[prediction] = 0.0
            vote_weights[prediction] += weight
            participating_agents.append(agent_name)

        # Handle no votes
        if not vote_weights:
            return VoteResult(
                consensus="ABSTAIN",
                confidence=0.0,
                vote_breakdown={},
                participating_agents=[],
                total_weight=0.0,
                agreement_score=0.0,
            )

        # Find consensus (highest weight)
        total_weight = sum(vote_weights.values())
        consensus = max(vote_weights, key=lambda k: vote_weights[k])
        consensus_weight = vote_weights[consensus]

        # Calculate confidence and agreement
        confidence = consensus_weight / total_weight if total_weight > 0 else 0.0
        agreement_score = confidence  # How unified the vote was

        # Normalize breakdown to percentages
        vote_breakdown = {k: v / total_weight if total_weight > 0 else 0.0 for k, v in vote_weights.items()}

        return VoteResult(
            consensus=consensus,
            confidence=confidence,
            vote_breakdown=vote_breakdown,
            participating_agents=participating_agents,
            total_weight=total_weight,
            agreement_score=agreement_score,
        )


class ContestManager:
    """
    Central manager for agent performance contest.

    Orchestrates:
    - Agent registration and tracking
    - Prediction recording
    - Outcome resolution
    - ELO updates
    - Weighted voting
    - Persistence
    """

    def __init__(
        self,
        elo_system: ELOSystem | None = None,
        voting_engine: VotingEngine | None = None,
        storage_dir: str = "contest_data",
        auto_persist: bool = True,
    ):
        """
        Initialize contest manager.

        Args:
            elo_system: ELO rating system
            voting_engine: Voting engine for consensus
            storage_dir: Directory for persistence
            auto_persist: Auto-save after updates
        """
        self.elo_system = elo_system or ELOSystem()
        self.voting_engine = voting_engine or VotingEngine()
        self.storage_dir = storage_dir
        self.auto_persist = auto_persist

        # Agent records
        self._agents: dict[str, AgentRecord] = {}

        # Pending predictions by symbol
        self._pending: dict[str, list[Prediction]] = {}

        # Prediction ID counter
        self._prediction_counter = 0

        # Ensure storage directory
        if auto_persist:
            os.makedirs(storage_dir, exist_ok=True)

    def register_agent(
        self,
        agent_name: str,
        initial_rating: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRecord:
        """
        Register a new agent in the contest.

        Args:
            agent_name: Unique agent identifier
            initial_rating: Starting ELO (defaults to base)
            metadata: Additional agent metadata

        Returns:
            New or existing AgentRecord
        """
        if agent_name in self._agents:
            return self._agents[agent_name]

        record = AgentRecord(
            agent_name=agent_name,
            elo_rating=initial_rating or self.elo_system.base_rating,
            metadata=metadata or {},
        )
        self._agents[agent_name] = record

        if self.auto_persist:
            self._persist()

        return record

    def get_agent(self, agent_name: str) -> AgentRecord | None:
        """Get agent record by name."""
        return self._agents.get(agent_name)

    def track_prediction(
        self,
        agent_name: str,
        prediction: str,
        confidence: float,
        symbol: str,
        metadata: dict[str, Any] | None = None,
    ) -> Prediction:
        """
        Track a new prediction from an agent.

        Args:
            agent_name: Name of predicting agent
            prediction: The prediction (e.g., "BUY", "SELL", "HOLD")
            confidence: Confidence level (0-1)
            symbol: Symbol being predicted
            metadata: Additional context

        Returns:
            Prediction record
        """
        # Auto-register agent if needed
        if agent_name not in self._agents:
            self.register_agent(agent_name)

        # Create prediction
        self._prediction_counter += 1
        pred = Prediction(
            prediction_id=f"pred_{self._prediction_counter:06d}",
            agent_name=agent_name,
            prediction=prediction,
            confidence=confidence,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )

        # Add to agent's record
        self._agents[agent_name].add_prediction(pred)

        # Add to pending for symbol
        if symbol not in self._pending:
            self._pending[symbol] = []
        self._pending[symbol].append(pred)

        if self.auto_persist:
            self._persist()

        return pred

    def record_outcome(
        self,
        symbol: str,
        actual: str,
        partial_matches: list[str] | None = None,
    ) -> list[tuple[str, PredictionOutcome, float]]:
        """
        Record the actual outcome for a symbol.

        Updates all pending predictions for this symbol.

        Args:
            symbol: The symbol
            actual: The actual outcome
            partial_matches: Predictions that count as partial matches

        Returns:
            List of (agent_name, outcome, new_rating)
        """
        partial_matches = partial_matches or []
        results = []

        pending = self._pending.pop(symbol, [])

        for pred in pending:
            # Determine outcome
            if pred.prediction == actual:
                outcome = PredictionOutcome.CORRECT
            elif pred.prediction in partial_matches:
                outcome = PredictionOutcome.PARTIAL
            else:
                outcome = PredictionOutcome.INCORRECT

            # Update prediction
            pred.outcome = outcome
            pred.actual = actual

            # Update agent record
            record = self._agents[pred.agent_name]
            record.record_outcome(outcome)

            # Update ELO
            new_rating = self.elo_system.update_from_prediction(
                record=record,
                outcome=outcome,
                confidence=pred.confidence,
            )

            results.append((pred.agent_name, outcome, new_rating))

        if self.auto_persist:
            self._persist()

        return results

    def get_rankings(self, limit: int = 10) -> list[AgentRanking]:
        """
        Get agent rankings by ELO rating.

        Args:
            limit: Maximum number to return

        Returns:
            List of AgentRanking sorted by ELO
        """
        # Sort agents by ELO
        sorted_agents = sorted(
            self._agents.values(),
            key=lambda a: a.elo_rating,
            reverse=True,
        )

        rankings = []
        for i, agent in enumerate(sorted_agents[:limit], 1):
            rankings.append(
                AgentRanking(
                    rank=i,
                    agent_name=agent.agent_name,
                    elo_rating=agent.elo_rating,
                    win_rate=agent.win_rate,
                    accuracy=agent.accuracy,
                    total_predictions=agent.total_predictions,
                    calibration_error=agent.calibration_error,
                )
            )

        return rankings

    def weighted_vote(
        self,
        predictions: dict[str, tuple[str, float]],
    ) -> VoteResult:
        """
        Perform weighted vote across agents.

        Args:
            predictions: Dict of agent_name -> (prediction, confidence)

        Returns:
            VoteResult with consensus
        """
        return self.voting_engine.weighted_vote(predictions, self._agents)

    def compare_agents(
        self,
        agent_a: str,
        agent_b: str,
    ) -> dict[str, Any]:
        """
        Compare two agents.

        Args:
            agent_a: First agent name
            agent_b: Second agent name

        Returns:
            Comparison dictionary
        """
        record_a = self._agents.get(agent_a)
        record_b = self._agents.get(agent_b)

        if not record_a or not record_b:
            return {"error": "Agent not found"}

        # Calculate expected outcomes
        expected_a = self.elo_system.expected_score(
            record_a.elo_rating,
            record_b.elo_rating,
        )

        return {
            "agent_a": {
                "name": agent_a,
                "elo": record_a.elo_rating,
                "accuracy": record_a.accuracy,
                "predictions": record_a.total_predictions,
            },
            "agent_b": {
                "name": agent_b,
                "elo": record_b.elo_rating,
                "accuracy": record_b.accuracy,
                "predictions": record_b.total_predictions,
            },
            "elo_difference": record_a.elo_rating - record_b.elo_rating,
            "expected_win_rate_a": expected_a,
            "expected_win_rate_b": 1 - expected_a,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get contest statistics."""
        if not self._agents:
            return {
                "total_agents": 0,
                "total_predictions": 0,
                "average_elo": 0.0,
                "average_accuracy": 0.0,
            }

        total_predictions = sum(a.total_predictions for a in self._agents.values())
        avg_elo = sum(a.elo_rating for a in self._agents.values()) / len(self._agents)
        accuracies = [a.accuracy for a in self._agents.values() if a.total_predictions > 0]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

        return {
            "total_agents": len(self._agents),
            "total_predictions": total_predictions,
            "pending_outcomes": sum(len(p) for p in self._pending.values()),
            "average_elo": round(avg_elo, 1),
            "average_accuracy": round(avg_accuracy, 3),
            "highest_elo": max(a.elo_rating for a in self._agents.values()),
            "lowest_elo": min(a.elo_rating for a in self._agents.values()),
        }

    def get_elo_history(
        self,
        agent_names: list[str] | None = None,
        limit: int = 100,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get ELO history for agents.

        Args:
            agent_names: List of agent names (None = all agents)
            limit: Maximum history entries per agent

        Returns:
            Dictionary mapping agent names to ELO history entries
        """
        if agent_names is None:
            agent_names = list(self._agents.keys())

        result: dict[str, list[dict[str, Any]]] = {}

        for name in agent_names:
            record = self._agents.get(name)
            if record is None:
                continue

            history = record.elo_history[-limit:] if record.elo_history else []
            result[name] = [entry.to_dict() for entry in history]

        return result

    def _persist(self) -> bool:
        """Persist contest data to storage."""
        try:
            filepath = os.path.join(self.storage_dir, "contest_state.json")
            data = {
                "agents": {name: record.to_dict() for name, record in self._agents.items()},
                "prediction_counter": self._prediction_counter,
                "saved_at": datetime.utcnow().isoformat(),
            }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            # P1 FIX: Log errors instead of silent failure
            logger.error(
                "CONTEST_PERSIST_FAILED",
                extra={"error": str(e), "filepath": filepath if "filepath" in dir() else "unknown"},
            )
            return False

    def load(self) -> bool:
        """Load contest data from storage."""
        try:
            filepath = os.path.join(self.storage_dir, "contest_state.json")
            if not os.path.exists(filepath):
                return False

            with open(filepath) as f:
                data = json.load(f)

            self._agents = {name: AgentRecord.from_dict(record) for name, record in data.get("agents", {}).items()}
            self._prediction_counter = data.get("prediction_counter", 0)
            return True
        except Exception as e:
            # P1 FIX: Log errors instead of silent failure
            logger.error(
                "CONTEST_LOAD_FAILED",
                extra={"error": str(e), "filepath": filepath if "filepath" in dir() else "unknown"},
            )
            return False


def create_contest_manager(
    k_factor: float = 32.0,
    min_confidence: float = 0.5,
    elo_weight_factor: float = 0.3,
    storage_dir: str = "contest_data",
    auto_persist: bool = True,
) -> ContestManager:
    """
    Factory function to create a ContestManager.

    Args:
        k_factor: ELO K-factor (rating volatility)
        min_confidence: Minimum confidence for voting
        elo_weight_factor: How much ELO affects votes
        storage_dir: Directory for persistence
        auto_persist: Auto-save after updates

    Returns:
        Configured ContestManager
    """
    elo_system = ELOSystem(k_factor=k_factor)
    voting_engine = VotingEngine(
        min_confidence=min_confidence,
        elo_weight_factor=elo_weight_factor,
    )

    return ContestManager(
        elo_system=elo_system,
        voting_engine=voting_engine,
        storage_dir=storage_dir,
        auto_persist=auto_persist,
    )
