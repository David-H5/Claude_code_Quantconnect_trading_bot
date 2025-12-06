"""
Agent Decision Logger

Provides comprehensive audit trails for all AI agent decisions.
Enables analysis, debugging, and compliance tracking.

Features:
- Structured decision logs with full context
- Reasoning chain capture
- Risk assessment tracking
- Alternatives considered logging
- Persistence to Object Store or file
- Decision pattern analysis

QuantConnect Compatible: Yes
- Non-blocking design
- Configurable storage backends
- Batch persistence support
"""

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol


class DecisionType(Enum):
    """Types of agent decisions."""

    TRADE = "trade"
    RISK_ASSESSMENT = "risk_assessment"
    STRATEGY_SELECTION = "strategy_selection"
    POSITION_SIZING = "position_sizing"
    EXIT_SIGNAL = "exit_signal"
    ALERT = "alert"
    ANALYSIS = "analysis"
    OTHER = "other"


class DecisionOutcome(Enum):
    """Outcome of a decision (for tracking)."""

    PENDING = "pending"
    EXECUTED = "executed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class RiskLevel(Enum):
    """Risk level assessment."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""

    step_number: int
    thought: str
    evidence: str | None = None
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Alternative:
    """An alternative considered but not chosen."""

    description: str
    reason_rejected: str
    estimated_outcome: str | None = None
    risk_level: RiskLevel = RiskLevel.MEDIUM


@dataclass
class RiskAssessment:
    """Risk assessment for a decision."""

    overall_level: RiskLevel
    factors: list[str]
    mitigation_steps: list[str]
    worst_case_scenario: str
    probability_of_loss: float = 0.0


@dataclass
class AgentDecisionLog:
    """
    Comprehensive log entry for an agent decision.

    Captures all context needed for audit, analysis, and debugging.
    """

    # Identity
    log_id: str
    timestamp: datetime
    agent_name: str
    agent_role: str

    # Decision
    decision_type: DecisionType
    decision: str
    confidence: float

    # Context
    context: dict[str, Any]
    query: str

    # Reasoning
    reasoning_chain: list[ReasoningStep]
    final_reasoning: str

    # Alternatives
    alternatives_considered: list[Alternative]

    # Risk
    risk_assessment: RiskAssessment

    # Execution
    execution_time_ms: float
    outcome: DecisionOutcome = DecisionOutcome.PENDING

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Sprint 1.5: Link to ReasoningLogger chain
    reasoning_chain_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "decision_type": self.decision_type.value,
            "decision": self.decision,
            "confidence": self.confidence,
            "context": self.context,
            "query": self.query,
            "reasoning_chain": [
                {
                    "step_number": r.step_number,
                    "thought": r.thought,
                    "evidence": r.evidence,
                    "confidence": r.confidence,
                }
                for r in self.reasoning_chain
            ],
            "final_reasoning": self.final_reasoning,
            "alternatives_considered": [
                {
                    "description": a.description,
                    "reason_rejected": a.reason_rejected,
                    "risk_level": a.risk_level.value,
                }
                for a in self.alternatives_considered
            ],
            "risk_assessment": {
                "overall_level": self.risk_assessment.overall_level.value,
                "factors": self.risk_assessment.factors,
                "mitigation_steps": self.risk_assessment.mitigation_steps,
                "worst_case_scenario": self.risk_assessment.worst_case_scenario,
                "probability_of_loss": self.risk_assessment.probability_of_loss,
            },
            "execution_time_ms": self.execution_time_ms,
            "outcome": self.outcome.value,
            "metadata": self.metadata,
            "reasoning_chain_id": self.reasoning_chain_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentDecisionLog":
        """Create from dictionary."""
        return cls(
            log_id=data["log_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            agent_name=data["agent_name"],
            agent_role=data["agent_role"],
            decision_type=DecisionType(data["decision_type"]),
            decision=data["decision"],
            confidence=data["confidence"],
            context=data["context"],
            query=data["query"],
            reasoning_chain=[
                ReasoningStep(
                    step_number=r["step_number"],
                    thought=r["thought"],
                    evidence=r.get("evidence"),
                    confidence=r.get("confidence", 0.5),
                )
                for r in data["reasoning_chain"]
            ],
            final_reasoning=data["final_reasoning"],
            alternatives_considered=[
                Alternative(
                    description=a["description"],
                    reason_rejected=a["reason_rejected"],
                    risk_level=RiskLevel(a.get("risk_level", "medium")),
                )
                for a in data["alternatives_considered"]
            ],
            risk_assessment=RiskAssessment(
                overall_level=RiskLevel(data["risk_assessment"]["overall_level"]),
                factors=data["risk_assessment"]["factors"],
                mitigation_steps=data["risk_assessment"]["mitigation_steps"],
                worst_case_scenario=data["risk_assessment"]["worst_case_scenario"],
                probability_of_loss=data["risk_assessment"].get("probability_of_loss", 0.0),
            ),
            execution_time_ms=data["execution_time_ms"],
            outcome=DecisionOutcome(data.get("outcome", "pending")),
            metadata=data.get("metadata", {}),
            reasoning_chain_id=data.get("reasoning_chain_id"),
        )


class StorageBackend(Protocol):
    """Protocol for storage backends."""

    def save(self, log: AgentDecisionLog) -> bool:
        """Save a decision log."""
        ...

    def load(self, log_id: str) -> AgentDecisionLog | None:
        """Load a decision log by ID."""
        ...

    def query(self, filters: dict[str, Any]) -> list[AgentDecisionLog]:
        """Query logs by filters."""
        ...


class InMemoryStorage:
    """In-memory storage for testing and development."""

    def __init__(self, max_size: int = 10000):
        self.logs: dict[str, AgentDecisionLog] = {}
        self.max_size = max_size

    def save(self, log: AgentDecisionLog) -> bool:
        """Save log to memory."""
        if len(self.logs) >= self.max_size:
            # Remove oldest
            oldest_id = min(self.logs.keys(), key=lambda k: self.logs[k].timestamp)
            del self.logs[oldest_id]

        self.logs[log.log_id] = log
        return True

    def load(self, log_id: str) -> AgentDecisionLog | None:
        """Load log by ID."""
        return self.logs.get(log_id)

    def query(self, filters: dict[str, Any]) -> list[AgentDecisionLog]:
        """Query logs by filters."""
        results = []
        for log in self.logs.values():
            match = True
            for key, value in filters.items():
                if hasattr(log, key):
                    if getattr(log, key) != value:
                        match = False
                        break
            if match:
                results.append(log)
        return results


class FileStorage:
    """File-based storage for persistence."""

    def __init__(self, directory: str = "decision_logs"):
        self.directory = directory
        import os

        os.makedirs(directory, exist_ok=True)

    def save(self, log: AgentDecisionLog) -> bool:
        """Save log to file."""
        try:
            import os

            filepath = os.path.join(self.directory, f"{log.log_id}.json")
            with open(filepath, "w") as f:
                json.dump(log.to_dict(), f, indent=2)
            return True
        except Exception:
            return False

    def load(self, log_id: str) -> AgentDecisionLog | None:
        """Load log from file."""
        try:
            import os

            filepath = os.path.join(self.directory, f"{log_id}.json")
            if not os.path.exists(filepath):
                return None
            with open(filepath) as f:
                data = json.load(f)
            return AgentDecisionLog.from_dict(data)
        except Exception:
            return None

    def query(self, filters: dict[str, Any]) -> list[AgentDecisionLog]:
        """Query logs from files (inefficient, use sparingly)."""
        import os

        results = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".json"):
                log = self.load(filename.replace(".json", ""))
                if log:
                    match = True
                    for key, value in filters.items():
                        if hasattr(log, key) and getattr(log, key) != value:
                            match = False
                            break
                    if match:
                        results.append(log)
        return results


@dataclass
class DecisionPatternAnalysis:
    """Analysis of decision patterns."""

    total_decisions: int
    decisions_by_type: dict[str, int]
    decisions_by_agent: dict[str, int]
    decisions_by_outcome: dict[str, int]
    average_confidence: float
    average_execution_time_ms: float
    high_risk_decisions: int
    low_confidence_decisions: int
    reasoning_chain_avg_length: float


class DecisionLogger:
    """
    Comprehensive agent decision audit trail.

    Logs all agent decisions with full context for:
    - Compliance and audit requirements
    - Performance analysis
    - Debugging and troubleshooting
    - Pattern identification

    Usage:
        logger = DecisionLogger()

        # Log a decision
        log = logger.log_decision(
            agent_name="technical_analyst",
            agent_role="analyst",
            decision_type=DecisionType.ANALYSIS,
            decision="BUY signal detected",
            confidence=0.85,
            context={"symbol": "SPY", "price": 450.0},
            query="Analyze SPY for entry opportunity",
            reasoning_chain=[...],
            risk_assessment=RiskAssessment(...),
        )

        # Query decisions
        spy_decisions = logger.get_decisions_by_context("symbol", "SPY")
    """

    def __init__(
        self,
        storage: StorageBackend | None = None,
        auto_persist: bool = True,
        batch_size: int = 100,
    ):
        """
        Initialize decision logger.

        Args:
            storage: Storage backend (defaults to in-memory)
            auto_persist: Automatically persist logs
            batch_size: Batch size for persistence
        """
        self.storage = storage or InMemoryStorage()
        self.auto_persist = auto_persist
        self.batch_size = batch_size

        # In-memory buffer
        self.logs: list[AgentDecisionLog] = []
        self._pending_persist: list[AgentDecisionLog] = []

    def log_decision(
        self,
        agent_name: str,
        agent_role: str,
        decision_type: DecisionType,
        decision: str,
        confidence: float,
        context: dict[str, Any],
        query: str,
        reasoning_chain: list[ReasoningStep],
        risk_assessment: RiskAssessment,
        final_reasoning: str = "",
        alternatives: list[Alternative] | None = None,
        execution_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
        reasoning_chain_id: str | None = None,
    ) -> AgentDecisionLog:
        """
        Log an agent decision.

        Args:
            agent_name: Name of the agent making the decision
            agent_role: Role of the agent
            decision_type: Type of decision
            decision: The decision made
            confidence: Confidence level (0-1)
            context: Full context for the decision
            query: Original query/task
            reasoning_chain: Steps in reasoning
            risk_assessment: Risk assessment
            final_reasoning: Summary reasoning
            alternatives: Alternatives considered
            execution_time_ms: Time to make decision
            metadata: Additional metadata
            reasoning_chain_id: Sprint 1.5 - Link to ReasoningLogger chain

        Returns:
            AgentDecisionLog entry
        """
        log_id = self._generate_log_id(agent_name, decision)

        log = AgentDecisionLog(
            log_id=log_id,
            timestamp=datetime.utcnow(),
            agent_name=agent_name,
            agent_role=agent_role,
            decision_type=decision_type,
            decision=decision,
            confidence=confidence,
            context=context,
            query=query,
            reasoning_chain=reasoning_chain,
            final_reasoning=final_reasoning or self._summarize_reasoning(reasoning_chain),
            alternatives_considered=alternatives or [],
            risk_assessment=risk_assessment,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
            reasoning_chain_id=reasoning_chain_id,
        )

        self.logs.append(log)

        if self.auto_persist:
            self._pending_persist.append(log)
            if len(self._pending_persist) >= self.batch_size:
                self._flush_pending()

        return log

    def update_outcome(
        self,
        log_id: str,
        outcome: DecisionOutcome,
        metadata_update: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update the outcome of a logged decision.

        Args:
            log_id: ID of the log to update
            outcome: New outcome
            metadata_update: Additional metadata to add

        Returns:
            True if updated successfully
        """
        # Update in memory
        for log in self.logs:
            if log.log_id == log_id:
                log.outcome = outcome
                if metadata_update:
                    log.metadata.update(metadata_update)
                # Re-persist
                if self.auto_persist:
                    self.storage.save(log)
                return True

        # Try storage
        log = self.storage.load(log_id)
        if log:
            log.outcome = outcome
            if metadata_update:
                log.metadata.update(metadata_update)
            return self.storage.save(log)

        return False

    def get_decisions_by_agent(self, agent_name: str) -> list[AgentDecisionLog]:
        """Get all decisions made by a specific agent."""
        return [log for log in self.logs if log.agent_name == agent_name]

    def get_decisions_by_type(self, decision_type: DecisionType) -> list[AgentDecisionLog]:
        """Get all decisions of a specific type."""
        return [log for log in self.logs if log.decision_type == decision_type]

    def get_decisions_by_context(
        self,
        context_key: str,
        context_value: Any,
    ) -> list[AgentDecisionLog]:
        """Get decisions matching a context key-value pair."""
        return [log for log in self.logs if log.context.get(context_key) == context_value]

    def get_high_risk_decisions(self) -> list[AgentDecisionLog]:
        """Get all high or critical risk decisions."""
        return [log for log in self.logs if log.risk_assessment.overall_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]

    def get_low_confidence_decisions(
        self,
        threshold: float = 0.5,
    ) -> list[AgentDecisionLog]:
        """Get decisions below confidence threshold."""
        return [log for log in self.logs if log.confidence < threshold]

    def get_decisions_by_chain_id(
        self,
        reasoning_chain_id: str,
    ) -> list[AgentDecisionLog]:
        """
        Get decisions linked to a specific reasoning chain.

        Sprint 1.5: Link between DecisionLogger and ReasoningLogger.

        Args:
            reasoning_chain_id: ID from ReasoningLogger chain

        Returns:
            List of decisions linked to this chain
        """
        return [log for log in self.logs if log.reasoning_chain_id == reasoning_chain_id]

    def analyze_patterns(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> DecisionPatternAnalysis:
        """
        Analyze patterns in logged decisions.

        Args:
            start_time: Filter start time
            end_time: Filter end time

        Returns:
            Pattern analysis results
        """
        logs = self.logs

        # Filter by time if specified
        if start_time:
            logs = [log for log in logs if log.timestamp >= start_time]
        if end_time:
            logs = [log for log in logs if log.timestamp <= end_time]

        if not logs:
            return DecisionPatternAnalysis(
                total_decisions=0,
                decisions_by_type={},
                decisions_by_agent={},
                decisions_by_outcome={},
                average_confidence=0.0,
                average_execution_time_ms=0.0,
                high_risk_decisions=0,
                low_confidence_decisions=0,
                reasoning_chain_avg_length=0.0,
            )

        # Count by type
        by_type: dict[str, int] = {}
        for log in logs:
            key = log.decision_type.value
            by_type[key] = by_type.get(key, 0) + 1

        # Count by agent
        by_agent: dict[str, int] = {}
        for log in logs:
            by_agent[log.agent_name] = by_agent.get(log.agent_name, 0) + 1

        # Count by outcome
        by_outcome: dict[str, int] = {}
        for log in logs:
            key = log.outcome.value
            by_outcome[key] = by_outcome.get(key, 0) + 1

        return DecisionPatternAnalysis(
            total_decisions=len(logs),
            decisions_by_type=by_type,
            decisions_by_agent=by_agent,
            decisions_by_outcome=by_outcome,
            average_confidence=statistics.mean(log.confidence for log in logs),
            average_execution_time_ms=statistics.mean(log.execution_time_ms for log in logs),
            high_risk_decisions=len(
                [log for log in logs if log.risk_assessment.overall_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
            ),
            low_confidence_decisions=len([log for log in logs if log.confidence < 0.5]),
            reasoning_chain_avg_length=statistics.mean(len(log.reasoning_chain) for log in logs) if logs else 0.0,
        )

    def _generate_log_id(self, agent_name: str, decision: str) -> str:
        """Generate unique log ID."""
        timestamp = datetime.utcnow().isoformat()
        content = f"{agent_name}:{decision}:{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _summarize_reasoning(self, chain: list[ReasoningStep]) -> str:
        """Summarize reasoning chain into final reasoning."""
        if not chain:
            return "No reasoning provided"
        return " -> ".join(step.thought for step in chain[-3:])  # Last 3 steps

    def _flush_pending(self) -> None:
        """Flush pending logs to storage."""
        for log in self._pending_persist:
            self.storage.save(log)
        self._pending_persist = []

    def flush(self) -> None:
        """Force flush all pending logs."""
        self._flush_pending()

    def clear(self) -> None:
        """Clear all logs (useful for testing)."""
        self.logs = []
        self._pending_persist = []


def create_decision_logger(
    storage_type: str = "memory",
    **kwargs,
) -> DecisionLogger:
    """
    Factory function to create a decision logger.

    Args:
        storage_type: "memory" or "file"
        **kwargs: Additional arguments for storage

    Returns:
        Configured DecisionLogger
    """
    if storage_type == "file":
        storage = FileStorage(kwargs.get("directory", "decision_logs"))
    else:
        storage = InMemoryStorage(kwargs.get("max_size", 10000))

    return DecisionLogger(storage=storage)


def generate_decision_report(
    logger: DecisionLogger,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> str:
    """
    Generate a report from decision logs.

    Args:
        logger: DecisionLogger instance
        start_time: Filter start
        end_time: Filter end

    Returns:
        Formatted report string
    """
    analysis = logger.analyze_patterns(start_time, end_time)

    lines = [
        "=" * 60,
        "AGENT DECISION LOG REPORT",
        "=" * 60,
        "",
        f"Total Decisions: {analysis.total_decisions}",
        f"Average Confidence: {analysis.average_confidence:.2f}",
        f"Average Execution Time: {analysis.average_execution_time_ms:.2f}ms",
        f"High Risk Decisions: {analysis.high_risk_decisions}",
        f"Low Confidence Decisions: {analysis.low_confidence_decisions}",
        "",
        "DECISIONS BY TYPE",
        "-" * 40,
    ]

    for dtype, count in sorted(analysis.decisions_by_type.items()):
        lines.append(f"  {dtype}: {count}")

    lines.extend(
        [
            "",
            "DECISIONS BY AGENT",
            "-" * 40,
        ]
    )

    for agent, count in sorted(analysis.decisions_by_agent.items()):
        lines.append(f"  {agent}: {count}")

    lines.extend(
        [
            "",
            "DECISIONS BY OUTCOME",
            "-" * 40,
        ]
    )

    for outcome, count in sorted(analysis.decisions_by_outcome.items()):
        lines.append(f"  {outcome}: {count}")

    lines.extend(["", "=" * 60])

    return "\n".join(lines)
