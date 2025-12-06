"""
Chain-of-Thought Reasoning Logger (UPGRADE-010 Sprint 1)

Provides unified reasoning chain tracking for all AI agent decisions.
Builds on DecisionLogger to add:
- ReasoningChain aggregation
- Cross-agent reasoning search
- Compliance audit trail export
- Historical reasoning queries

Part of UPGRADE-010: Advanced AI Features
Phase: 1 (Explainable AI)

QuantConnect Compatible: Yes
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Import existing infrastructure
from llm.decision_logger import (
    DecisionLogger,
    ReasoningStep,
)


class ChainStatus(Enum):
    """Status of a reasoning chain."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    FAILED = "failed"


@dataclass
class ReasoningChain:
    """
    Aggregates reasoning steps into a complete chain.

    Combines ReasoningStep instances from decision_logger with
    additional metadata for tracking multi-step reasoning processes.
    """

    chain_id: str
    agent_name: str
    task: str
    started_at: datetime
    steps: list[ReasoningStep] = field(default_factory=list)
    status: ChainStatus = ChainStatus.IN_PROGRESS
    completed_at: datetime | None = None
    final_decision: str | None = None
    final_confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        thought: str,
        evidence: str | None = None,
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> ReasoningStep:
        """Add a reasoning step to the chain."""
        step = ReasoningStep(
            step_number=len(self.steps) + 1,
            thought=thought,
            evidence=evidence,
            confidence=confidence,
            metadata=metadata or {},
        )
        self.steps.append(step)
        return step

    def complete(
        self,
        decision: str,
        confidence: float,
    ) -> None:
        """Mark the chain as completed with final decision."""
        self.status = ChainStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.final_decision = decision
        self.final_confidence = confidence

    def fail(self, reason: str) -> None:
        """Mark the chain as failed."""
        self.status = ChainStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.metadata["failure_reason"] = reason

    def abandon(self, reason: str) -> None:
        """Mark the chain as abandoned."""
        self.status = ChainStatus.ABANDONED
        self.completed_at = datetime.utcnow()
        self.metadata["abandon_reason"] = reason

    @property
    def duration_ms(self) -> float:
        """Get chain duration in milliseconds."""
        if self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds() * 1000
        return 0.0

    @property
    def average_confidence(self) -> float:
        """Get average confidence across all steps."""
        if not self.steps:
            return 0.0
        return sum(s.confidence for s in self.steps) / len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chain_id": self.chain_id,
            "agent_name": self.agent_name,
            "task": self.task,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "steps": [
                {
                    "step_number": s.step_number,
                    "thought": s.thought,
                    "evidence": s.evidence,
                    "confidence": s.confidence,
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
            "final_decision": self.final_decision,
            "final_confidence": self.final_confidence,
            "duration_ms": self.duration_ms,
            "average_confidence": self.average_confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReasoningChain":
        """Create from dictionary."""
        chain = cls(
            chain_id=data["chain_id"],
            agent_name=data["agent_name"],
            task=data["task"],
            started_at=datetime.fromisoformat(data["started_at"]),
            status=ChainStatus(data["status"]),
            final_decision=data.get("final_decision"),
            final_confidence=data.get("final_confidence", 0.0),
            metadata=data.get("metadata", {}),
        )
        if data.get("completed_at"):
            chain.completed_at = datetime.fromisoformat(data["completed_at"])

        for step_data in data.get("steps", []):
            chain.steps.append(
                ReasoningStep(
                    step_number=step_data["step_number"],
                    thought=step_data["thought"],
                    evidence=step_data.get("evidence"),
                    confidence=step_data.get("confidence", 0.5),
                    metadata=step_data.get("metadata", {}),
                )
            )

        return chain


@dataclass
class SearchResult:
    """Result from reasoning search."""

    chain_id: str
    agent_name: str
    task: str
    matching_steps: list[ReasoningStep]
    relevance_score: float
    timestamp: datetime


@dataclass
class AuditTrailEntry:
    """Single entry in compliance audit trail."""

    timestamp: str
    agent_name: str
    task: str
    chain_id: str
    step_count: int
    final_decision: str | None
    confidence: float
    duration_ms: float
    status: str


class ReasoningLogger:
    """
    Unified reasoning chain logger for AI agents.

    Provides:
    - Chain creation and tracking
    - Step-by-step reasoning logging
    - Cross-chain search
    - Compliance audit trail export
    - Integration with DecisionLogger

    Usage:
        logger = ReasoningLogger()

        # Start a reasoning chain
        chain = logger.start_chain("technical_analyst", "Analyze SPY entry")

        # Add reasoning steps
        chain.add_step("RSI at 35, oversold territory", confidence=0.8)
        chain.add_step("Price above 200 SMA", evidence="SMA(200)=445", confidence=0.9)
        chain.add_step("Volume confirming uptrend", confidence=0.7)

        # Complete with decision
        chain.complete("BUY", confidence=0.85)

        # Query historical reasoning
        results = logger.search_reasoning("oversold")

        # Export for compliance
        logger.export_audit_trail("audit_2025.json")
    """

    def __init__(
        self,
        storage_dir: str = "reasoning_chains",
        decision_logger: DecisionLogger | None = None,
        auto_persist: bool = True,
    ):
        """
        Initialize reasoning logger.

        Args:
            storage_dir: Directory for chain persistence
            decision_logger: Optional DecisionLogger for integration
            auto_persist: Automatically save chains on completion
        """
        self.storage_dir = storage_dir
        self.decision_logger = decision_logger
        self.auto_persist = auto_persist

        # In-memory chain storage
        self._active_chains: dict[str, ReasoningChain] = {}
        self._completed_chains: list[ReasoningChain] = []

        # Ensure storage directory exists
        if auto_persist:
            os.makedirs(storage_dir, exist_ok=True)

    def start_chain(
        self,
        agent_name: str,
        task: str,
        metadata: dict[str, Any] | None = None,
    ) -> ReasoningChain:
        """
        Start a new reasoning chain.

        Args:
            agent_name: Name of the agent starting the chain
            task: Description of the task/query
            metadata: Additional context

        Returns:
            New ReasoningChain
        """
        chain_id = self._generate_chain_id(agent_name, task)

        chain = ReasoningChain(
            chain_id=chain_id,
            agent_name=agent_name,
            task=task,
            started_at=datetime.utcnow(),
            metadata=metadata or {},
        )

        self._active_chains[chain_id] = chain
        return chain

    def complete_chain(
        self,
        chain_id: str,
        decision: str,
        confidence: float,
    ) -> ReasoningChain | None:
        """
        Complete an active reasoning chain.

        Args:
            chain_id: ID of the chain to complete
            decision: Final decision
            confidence: Confidence in decision

        Returns:
            Completed chain or None if not found
        """
        chain = self._active_chains.pop(chain_id, None)
        if not chain:
            return None

        chain.complete(decision, confidence)
        self._completed_chains.append(chain)

        if self.auto_persist:
            self._persist_chain(chain)

        return chain

    def get_chain(self, chain_id: str) -> ReasoningChain | None:
        """Get a chain by ID (active or completed)."""
        # Check active first
        if chain_id in self._active_chains:
            return self._active_chains[chain_id]

        # Check completed
        for chain in self._completed_chains:
            if chain.chain_id == chain_id:
                return chain

        # Try to load from storage
        return self._load_chain(chain_id)

    def get_chains_by_agent(self, agent_name: str) -> list[ReasoningChain]:
        """Get all chains for a specific agent."""
        chains = []

        # Active chains
        for chain in self._active_chains.values():
            if chain.agent_name == agent_name:
                chains.append(chain)

        # Completed chains
        for chain in self._completed_chains:
            if chain.agent_name == agent_name:
                chains.append(chain)

        return chains

    def search_reasoning(
        self,
        query: str,
        agent_name: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Search through reasoning chains for matching content.

        Args:
            query: Search query (substring match)
            agent_name: Filter by agent name
            limit: Maximum results to return

        Returns:
            List of matching SearchResults
        """
        results = []
        query_lower = query.lower()

        # Search all chains
        all_chains = list(self._active_chains.values()) + self._completed_chains

        for chain in all_chains:
            # Filter by agent if specified
            if agent_name and chain.agent_name != agent_name:
                continue

            # Search steps
            matching_steps = []
            for step in chain.steps:
                thought_match = query_lower in step.thought.lower()
                evidence_match = step.evidence and query_lower in step.evidence.lower()

                if thought_match or evidence_match:
                    matching_steps.append(step)

            # Also check task
            task_match = query_lower in chain.task.lower()

            if matching_steps or task_match:
                # Calculate relevance score
                relevance = len(matching_steps) / max(len(chain.steps), 1)
                if task_match:
                    relevance += 0.5

                results.append(
                    SearchResult(
                        chain_id=chain.chain_id,
                        agent_name=chain.agent_name,
                        task=chain.task,
                        matching_steps=matching_steps,
                        relevance_score=min(relevance, 1.0),
                        timestamp=chain.started_at,
                    )
                )

        # Sort by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        return results[:limit]

    def export_audit_trail(
        self,
        filepath: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        include_steps: bool = True,
    ) -> int:
        """
        Export reasoning chains as compliance audit trail.

        Args:
            filepath: Output file path
            start_time: Filter start time
            end_time: Filter end time
            include_steps: Include individual reasoning steps

        Returns:
            Number of chains exported
        """
        all_chains = list(self._active_chains.values()) + self._completed_chains

        # Filter by time
        if start_time:
            all_chains = [c for c in all_chains if c.started_at >= start_time]
        if end_time:
            all_chains = [c for c in all_chains if c.started_at <= end_time]

        # Sort by timestamp
        all_chains.sort(key=lambda c: c.started_at)

        # Build audit trail
        audit_trail = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_chains": len(all_chains),
            "filter_start": start_time.isoformat() if start_time else None,
            "filter_end": end_time.isoformat() if end_time else None,
            "entries": [],
        }

        for chain in all_chains:
            entry = {
                "timestamp": chain.started_at.isoformat(),
                "agent_name": chain.agent_name,
                "task": chain.task,
                "chain_id": chain.chain_id,
                "step_count": len(chain.steps),
                "final_decision": chain.final_decision,
                "confidence": chain.final_confidence,
                "duration_ms": chain.duration_ms,
                "status": chain.status.value,
            }

            if include_steps:
                entry["reasoning_steps"] = [
                    {
                        "step": s.step_number,
                        "thought": s.thought,
                        "evidence": s.evidence,
                        "confidence": s.confidence,
                    }
                    for s in chain.steps
                ]

            audit_trail["entries"].append(entry)

        # Write to file
        with open(filepath, "w") as f:
            json.dump(audit_trail, f, indent=2)

        return len(all_chains)

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about reasoning chains."""
        all_chains = list(self._active_chains.values()) + self._completed_chains
        completed = [c for c in all_chains if c.status == ChainStatus.COMPLETED]

        if not all_chains:
            return {
                "total_chains": 0,
                "active_chains": 0,
                "completed_chains": 0,
            }

        total_steps = sum(len(c.steps) for c in all_chains)
        avg_steps = total_steps / len(all_chains)

        avg_confidence = 0.0
        if completed:
            avg_confidence = sum(c.final_confidence for c in completed) / len(completed)

        avg_duration = 0.0
        if completed:
            avg_duration = sum(c.duration_ms for c in completed) / len(completed)

        # Chains by agent
        by_agent: dict[str, int] = {}
        for chain in all_chains:
            by_agent[chain.agent_name] = by_agent.get(chain.agent_name, 0) + 1

        return {
            "total_chains": len(all_chains),
            "active_chains": len(self._active_chains),
            "completed_chains": len(completed),
            "failed_chains": len([c for c in all_chains if c.status == ChainStatus.FAILED]),
            "abandoned_chains": len([c for c in all_chains if c.status == ChainStatus.ABANDONED]),
            "total_steps": total_steps,
            "average_steps_per_chain": round(avg_steps, 2),
            "average_confidence": round(avg_confidence, 3),
            "average_duration_ms": round(avg_duration, 2),
            "chains_by_agent": by_agent,
        }

    def _generate_chain_id(self, agent_name: str, task: str) -> str:
        """Generate unique chain ID."""
        import hashlib

        timestamp = datetime.utcnow().isoformat()
        content = f"{agent_name}:{task}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _persist_chain(self, chain: ReasoningChain) -> bool:
        """Persist a chain to storage."""
        try:
            filepath = os.path.join(self.storage_dir, f"{chain.chain_id}.json")
            with open(filepath, "w") as f:
                json.dump(chain.to_dict(), f, indent=2)
            return True
        except Exception:
            return False

    def _load_chain(self, chain_id: str) -> ReasoningChain | None:
        """Load a chain from storage."""
        try:
            filepath = os.path.join(self.storage_dir, f"{chain_id}.json")
            if not os.path.exists(filepath):
                return None
            with open(filepath) as f:
                data = json.load(f)
            return ReasoningChain.from_dict(data)
        except Exception:
            return None


def create_reasoning_logger(
    storage_dir: str = "reasoning_chains",
    decision_logger: DecisionLogger | None = None,
    auto_persist: bool = True,
) -> ReasoningLogger:
    """
    Factory function to create a ReasoningLogger.

    Args:
        storage_dir: Directory for chain persistence
        decision_logger: Optional DecisionLogger for integration
        auto_persist: Automatically save chains on completion

    Returns:
        Configured ReasoningLogger
    """
    return ReasoningLogger(
        storage_dir=storage_dir,
        decision_logger=decision_logger,
        auto_persist=auto_persist,
    )
