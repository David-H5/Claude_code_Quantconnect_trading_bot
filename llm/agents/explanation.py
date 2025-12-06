"""Agent Decision Explainability Framework.

Provides human-readable explanations for agent decisions.
Extends the DecisionLogger with formatted output for:
- Audit reports
- User interfaces
- Regulatory compliance
- Debugging

Usage:
    from llm.agents.explanation import ExplanationReport, create_explanation

    explanation = create_explanation(decision_log)
    print(explanation.to_human_readable())
    print(explanation.to_markdown())
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from llm.decision_logger import (
    AgentDecisionLog,
    Alternative,
    RiskAssessment,
    RiskLevel,
)


class ConfidenceLevel(Enum):
    """Human-readable confidence levels."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

    @classmethod
    def from_float(cls, value: float) -> "ConfidenceLevel":
        """Convert numeric confidence to level."""
        if value < 0.2:
            return cls.VERY_LOW
        if value < 0.4:
            return cls.LOW
        if value < 0.6:
            return cls.MEDIUM
        if value < 0.8:
            return cls.HIGH
        return cls.VERY_HIGH


@dataclass
class SignalContribution:
    """A signal that contributed to the decision.

    Represents one input source that influenced the agent's decision.
    """

    source: str  # e.g., "technical_analyst", "sentiment", "news"
    signal_type: str  # e.g., "bullish", "bearish", "neutral"
    weight: float  # 0.0 to 1.0
    reasoning: str
    confidence: ConfidenceLevel
    data_points: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "signal_type": self.signal_type,
            "weight": self.weight,
            "reasoning": self.reasoning,
            "confidence": self.confidence.value,
            "data_points": self.data_points,
        }


@dataclass
class AlternativeConsidered:
    """An alternative that was considered but not chosen.

    Wraps the base Alternative with additional explanation context.
    """

    action: str
    expected_outcome: str
    reason_rejected: str
    confidence_if_chosen: float
    risk_level: RiskLevel = RiskLevel.MEDIUM

    @classmethod
    def from_alternative(cls, alt: Alternative) -> "AlternativeConsidered":
        """Create from base Alternative."""
        return cls(
            action=alt.description,
            expected_outcome=alt.estimated_outcome or "Unknown",
            reason_rejected=alt.reason_rejected,
            confidence_if_chosen=0.0,  # Would need to be tracked separately
            risk_level=alt.risk_level,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action,
            "expected_outcome": self.expected_outcome,
            "reason_rejected": self.reason_rejected,
            "confidence_if_chosen": self.confidence_if_chosen,
            "risk_level": self.risk_level.value,
        }


@dataclass
class DetailedRiskAssessment:
    """Extended risk assessment with more detail for explanations.

    Includes financial context and justifications.
    """

    max_loss_estimate: float
    probability_of_loss: float
    risk_reward_ratio: float
    position_size_justification: str
    stop_loss_reasoning: str
    overall_level: RiskLevel = RiskLevel.MEDIUM
    factors: list[str] = field(default_factory=list)
    mitigation_steps: list[str] = field(default_factory=list)

    @classmethod
    def from_risk_assessment(
        cls, ra: RiskAssessment, max_loss: float = 0.0, rr_ratio: float = 0.0
    ) -> "DetailedRiskAssessment":
        """Create from base RiskAssessment."""
        return cls(
            max_loss_estimate=max_loss,
            probability_of_loss=ra.probability_of_loss,
            risk_reward_ratio=rr_ratio,
            position_size_justification="Based on risk limits",
            stop_loss_reasoning=ra.worst_case_scenario,
            overall_level=ra.overall_level,
            factors=ra.factors,
            mitigation_steps=ra.mitigation_steps,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_loss_estimate": self.max_loss_estimate,
            "probability_of_loss": self.probability_of_loss,
            "risk_reward_ratio": self.risk_reward_ratio,
            "position_size_justification": self.position_size_justification,
            "stop_loss_reasoning": self.stop_loss_reasoning,
            "overall_level": self.overall_level.value,
            "factors": self.factors,
            "mitigation_steps": self.mitigation_steps,
        }


@dataclass
class ExplanationReport:
    """Comprehensive explanation of an agent decision.

    Provides full transparency for audit, debugging, and regulatory compliance.
    Can be generated from an AgentDecisionLog.
    """

    # Metadata
    agent_name: str
    decision_id: str
    timestamp: datetime

    # The decision
    action: str  # "buy", "sell", "hold", "close"
    symbol: str
    quantity: int | None = None
    price_target: float | None = None

    # Reasoning
    reasoning_chain: list[str] = field(default_factory=list)
    contributing_signals: list[SignalContribution] = field(default_factory=list)

    # Confidence
    overall_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_breakdown: dict[str, float] = field(default_factory=dict)

    # Alternatives
    alternatives_considered: list[AlternativeConsidered] = field(default_factory=list)

    # Risk
    risk_assessment: DetailedRiskAssessment | None = None

    # Context
    market_context: dict[str, Any] = field(default_factory=dict)
    position_context: dict[str, Any] = field(default_factory=dict)

    # Original log reference
    original_log_id: str | None = None

    def to_human_readable(self) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"## Decision Explanation: {self.action.upper()} {self.symbol}",
            f"**Agent:** {self.agent_name}",
            f"**Time:** {self.timestamp.isoformat()}",
            f"**Confidence:** {self.overall_confidence.value}",
            "",
        ]

        if self.quantity:
            lines.append(f"**Quantity:** {self.quantity}")
        if self.price_target:
            lines.append(f"**Price Target:** ${self.price_target:.2f}")
        lines.append("")

        lines.append("### Reasoning Chain")
        for i, step in enumerate(self.reasoning_chain, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

        if self.contributing_signals:
            lines.append("### Contributing Signals")
            for signal in self.contributing_signals:
                lines.append(
                    f"- **{signal.source}** ({signal.signal_type}): "
                    f"{signal.reasoning} [weight: {signal.weight:.2f}]"
                )
            lines.append("")

        if self.alternatives_considered:
            lines.append("### Alternatives Considered")
            for alt in self.alternatives_considered:
                lines.append(f"- **{alt.action}**: {alt.reason_rejected}")
            lines.append("")

        if self.risk_assessment:
            lines.extend(
                [
                    "### Risk Assessment",
                    f"- Overall Risk: {self.risk_assessment.overall_level.value}",
                    f"- Max Loss Estimate: ${self.risk_assessment.max_loss_estimate:.2f}",
                    f"- Probability of Loss: {self.risk_assessment.probability_of_loss:.1%}",
                    f"- Risk/Reward: {self.risk_assessment.risk_reward_ratio:.2f}",
                    "",
                ]
            )

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        return self.to_human_readable()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_name": self.agent_name,
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "price_target": self.price_target,
            "reasoning_chain": self.reasoning_chain,
            "contributing_signals": [s.to_dict() for s in self.contributing_signals],
            "overall_confidence": self.overall_confidence.value,
            "confidence_breakdown": self.confidence_breakdown,
            "alternatives_considered": [a.to_dict() for a in self.alternatives_considered],
            "risk_assessment": self.risk_assessment.to_dict() if self.risk_assessment else None,
            "market_context": self.market_context,
            "position_context": self.position_context,
            "original_log_id": self.original_log_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExplanationReport":
        """Create from dictionary."""
        return cls(
            agent_name=data["agent_name"],
            decision_id=data["decision_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action=data["action"],
            symbol=data["symbol"],
            quantity=data.get("quantity"),
            price_target=data.get("price_target"),
            reasoning_chain=data.get("reasoning_chain", []),
            contributing_signals=[
                SignalContribution(
                    source=s["source"],
                    signal_type=s["signal_type"],
                    weight=s["weight"],
                    reasoning=s["reasoning"],
                    confidence=ConfidenceLevel(s["confidence"]),
                    data_points=s.get("data_points", []),
                )
                for s in data.get("contributing_signals", [])
            ],
            overall_confidence=ConfidenceLevel(data.get("overall_confidence", "medium")),
            confidence_breakdown=data.get("confidence_breakdown", {}),
            alternatives_considered=[
                AlternativeConsidered(
                    action=a["action"],
                    expected_outcome=a["expected_outcome"],
                    reason_rejected=a["reason_rejected"],
                    confidence_if_chosen=a.get("confidence_if_chosen", 0),
                    risk_level=RiskLevel(a.get("risk_level", "medium")),
                )
                for a in data.get("alternatives_considered", [])
            ],
            risk_assessment=(
                DetailedRiskAssessment(
                    max_loss_estimate=data["risk_assessment"]["max_loss_estimate"],
                    probability_of_loss=data["risk_assessment"]["probability_of_loss"],
                    risk_reward_ratio=data["risk_assessment"]["risk_reward_ratio"],
                    position_size_justification=data["risk_assessment"]["position_size_justification"],
                    stop_loss_reasoning=data["risk_assessment"]["stop_loss_reasoning"],
                    overall_level=RiskLevel(data["risk_assessment"]["overall_level"]),
                    factors=data["risk_assessment"].get("factors", []),
                    mitigation_steps=data["risk_assessment"].get("mitigation_steps", []),
                )
                if data.get("risk_assessment")
                else None
            ),
            market_context=data.get("market_context", {}),
            position_context=data.get("position_context", {}),
            original_log_id=data.get("original_log_id"),
        )


def create_explanation(log: AgentDecisionLog) -> ExplanationReport:
    """Create an ExplanationReport from an AgentDecisionLog.

    Args:
        log: The decision log to explain.

    Returns:
        Formatted explanation report.
    """
    # Extract action and symbol from decision
    decision_parts = log.decision.lower().split()
    action = decision_parts[0] if decision_parts else "unknown"
    symbol = log.context.get("symbol", "UNKNOWN")

    # Convert reasoning chain to strings
    reasoning_chain = [step.thought for step in log.reasoning_chain]
    if log.final_reasoning:
        reasoning_chain.append(f"Conclusion: {log.final_reasoning}")

    # Convert alternatives
    alternatives = [AlternativeConsidered.from_alternative(alt) for alt in log.alternatives_considered]

    # Convert risk assessment
    risk = None
    if log.risk_assessment:
        risk = DetailedRiskAssessment.from_risk_assessment(
            log.risk_assessment,
            max_loss=log.context.get("max_loss", 0.0),
            rr_ratio=log.context.get("risk_reward_ratio", 0.0),
        )

    return ExplanationReport(
        agent_name=log.agent_name,
        decision_id=log.log_id,
        timestamp=log.timestamp,
        action=action,
        symbol=symbol,
        quantity=log.context.get("quantity"),
        price_target=log.context.get("price"),
        reasoning_chain=reasoning_chain,
        overall_confidence=ConfidenceLevel.from_float(log.confidence),
        confidence_breakdown={log.agent_name: log.confidence},
        alternatives_considered=alternatives,
        risk_assessment=risk,
        market_context=log.context.get("market", {}),
        position_context=log.context.get("position", {}),
        original_log_id=log.log_id,
    )
