"""
Decision Tracer for AI Agent Reasoning

UPGRADE-015 Phase 6: Observability Setup

Traces and records AI agent decision-making processes for debugging,
analysis, and compliance purposes.

Features:
- Decision chain tracking
- Reasoning step recording
- Confidence scoring
- Decision outcome tracking
- Export for analysis
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class DecisionOutcome(Enum):
    """Possible decision outcomes."""

    PENDING = "pending"
    EXECUTED = "executed"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    ERROR = "error"
    OVERRIDDEN = "overridden"


class DecisionCategory(Enum):
    """Categories of decisions."""

    TRADE = "trade"
    RISK = "risk"
    POSITION = "position"
    STRATEGY = "strategy"
    EXECUTION = "execution"
    ANALYSIS = "analysis"


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""

    step_number: int
    description: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Decision:
    """Represents a traced decision."""

    decision_id: str
    category: DecisionCategory
    description: str
    agent_name: str
    reasoning_chain: list[ReasoningStep] = field(default_factory=list)
    final_confidence: float = 0.0
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    outcome_details: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class DecisionTracer:
    """Traces and records agent decisions."""

    def __init__(
        self,
        log_dir: str | Path | None = None,
        auto_persist: bool = True,
    ):
        """
        Initialize decision tracer.

        Args:
            log_dir: Directory for decision logs
            auto_persist: Whether to auto-save decisions
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs/decisions")
        self.auto_persist = auto_persist

        self._decisions: dict[str, Decision] = {}
        self._decision_counter = 0

        if auto_persist:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def start_decision(
        self,
        category: DecisionCategory,
        description: str,
        agent_name: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Start tracing a new decision.

        Args:
            category: Decision category
            description: Description of the decision
            agent_name: Name of the agent making the decision
            context: Initial context data

        Returns:
            Decision ID
        """
        self._decision_counter += 1
        decision_id = f"dec_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self._decision_counter}"

        decision = Decision(
            decision_id=decision_id,
            category=category,
            description=description,
            agent_name=agent_name,
            context=context or {},
        )

        self._decisions[decision_id] = decision
        return decision_id

    def add_reasoning_step(
        self,
        decision_id: str,
        description: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        confidence: float = 1.0,
    ) -> int:
        """
        Add a reasoning step to a decision.

        Args:
            decision_id: ID of the decision
            description: Description of the reasoning step
            inputs: Input data for the step
            outputs: Output data from the step
            confidence: Confidence score for this step

        Returns:
            Step number
        """
        if decision_id not in self._decisions:
            raise ValueError(f"Decision {decision_id} not found")

        decision = self._decisions[decision_id]
        step_number = len(decision.reasoning_chain) + 1

        step = ReasoningStep(
            step_number=step_number,
            description=description,
            inputs=inputs or {},
            outputs=outputs or {},
            confidence=confidence,
        )

        decision.reasoning_chain.append(step)
        return step_number

    def complete_decision(
        self,
        decision_id: str,
        outcome: DecisionOutcome,
        final_confidence: float,
        outcome_details: str | None = None,
    ) -> Decision:
        """
        Complete a decision and record the outcome.

        Args:
            decision_id: ID of the decision
            outcome: Final outcome
            final_confidence: Final confidence score
            outcome_details: Details about the outcome

        Returns:
            Completed decision
        """
        if decision_id not in self._decisions:
            raise ValueError(f"Decision {decision_id} not found")

        decision = self._decisions[decision_id]
        decision.outcome = outcome
        decision.final_confidence = final_confidence
        decision.outcome_details = outcome_details
        decision.completed_at = datetime.utcnow()

        if self.auto_persist:
            self._persist_decision(decision)

        return decision

    def get_decision(self, decision_id: str) -> Decision | None:
        """Get a decision by ID."""
        return self._decisions.get(decision_id)

    def get_decisions_by_category(
        self,
        category: DecisionCategory,
        limit: int = 100,
    ) -> list[Decision]:
        """Get decisions filtered by category."""
        decisions = [d for d in self._decisions.values() if d.category == category]
        return sorted(decisions, key=lambda d: d.created_at, reverse=True)[:limit]

    def get_decisions_by_outcome(
        self,
        outcome: DecisionOutcome,
        limit: int = 100,
    ) -> list[Decision]:
        """Get decisions filtered by outcome."""
        decisions = [d for d in self._decisions.values() if d.outcome == outcome]
        return sorted(decisions, key=lambda d: d.created_at, reverse=True)[:limit]

    def get_recent_decisions(self, limit: int = 50) -> list[Decision]:
        """Get most recent decisions."""
        decisions = list(self._decisions.values())
        return sorted(decisions, key=lambda d: d.created_at, reverse=True)[:limit]

    def _persist_decision(self, decision: Decision) -> None:
        """Persist a decision to disk."""
        try:
            file_path = self.log_dir / f"{decision.decision_id}.json"
            data = self._decision_to_dict(decision)
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to persist decision {decision.decision_id}: {e}")

    def _decision_to_dict(self, decision: Decision) -> dict[str, Any]:
        """Convert decision to dictionary for serialization."""
        return {
            "decision_id": decision.decision_id,
            "category": decision.category.value,
            "description": decision.description,
            "agent_name": decision.agent_name,
            "reasoning_chain": [
                {
                    "step_number": step.step_number,
                    "description": step.description,
                    "inputs": step.inputs,
                    "outputs": step.outputs,
                    "confidence": step.confidence,
                    "timestamp": step.timestamp.isoformat(),
                }
                for step in decision.reasoning_chain
            ],
            "final_confidence": decision.final_confidence,
            "outcome": decision.outcome.value,
            "outcome_details": decision.outcome_details,
            "created_at": decision.created_at.isoformat(),
            "completed_at": decision.completed_at.isoformat() if decision.completed_at else None,
            "context": decision.context,
            "metadata": decision.metadata,
        }

    def export_decisions(
        self,
        output_path: str | Path,
        decisions: list[Decision] | None = None,
    ) -> None:
        """
        Export decisions to a JSON file.

        Args:
            output_path: Path for the output file
            decisions: Decisions to export (or all if None)
        """
        if decisions is None:
            decisions = list(self._decisions.values())

        data = {
            "exported_at": datetime.utcnow().isoformat(),
            "total_decisions": len(decisions),
            "decisions": [self._decision_to_dict(d) for d in decisions],
        }

        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_decision_stats(self) -> dict[str, Any]:
        """
        Get statistics about traced decisions.

        Returns:
            Dictionary of statistics
        """
        decisions = list(self._decisions.values())

        if not decisions:
            return {
                "total_decisions": 0,
                "by_category": {},
                "by_outcome": {},
                "avg_confidence": 0.0,
                "avg_reasoning_steps": 0.0,
            }

        by_category = {}
        for cat in DecisionCategory:
            count = len([d for d in decisions if d.category == cat])
            if count > 0:
                by_category[cat.value] = count

        by_outcome = {}
        for outcome in DecisionOutcome:
            count = len([d for d in decisions if d.outcome == outcome])
            if count > 0:
                by_outcome[outcome.value] = count

        completed = [d for d in decisions if d.outcome != DecisionOutcome.PENDING]
        avg_confidence = sum(d.final_confidence for d in completed) / len(completed) if completed else 0.0
        avg_steps = sum(len(d.reasoning_chain) for d in decisions) / len(decisions) if decisions else 0.0

        return {
            "total_decisions": len(decisions),
            "by_category": by_category,
            "by_outcome": by_outcome,
            "avg_confidence": round(avg_confidence, 3),
            "avg_reasoning_steps": round(avg_steps, 1),
        }


def create_decision_tracer(
    log_dir: str | Path | None = None,
    auto_persist: bool = True,
) -> DecisionTracer:
    """
    Factory function to create a decision tracer.

    Args:
        log_dir: Directory for decision logs
        auto_persist: Whether to auto-save decisions

    Returns:
        Configured DecisionTracer
    """
    return DecisionTracer(log_dir=log_dir, auto_persist=auto_persist)
