"""
Sprint 1 Unified Monitoring Report (UPGRADE-010 Sprint 1.5)

Provides a unified monitoring report combining all Sprint 1 components:
- Reasoning chains from ReasoningLogger
- Anomaly history from AnomalyDetector
- Explanation audit trail from ExplainerLogger

Part of UPGRADE-010: Advanced AI Features
Phase: Sprint 1.5 Polish

QuantConnect Compatible: Yes
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from llm.reasoning_logger import ReasoningLogger
from models.anomaly_detector import AnomalyDetector


@dataclass
class ReasoningChainSummary:
    """Summary of reasoning chain statistics."""

    total_chains: int = 0
    completed_chains: int = 0
    active_chains: int = 0
    failed_chains: int = 0
    average_steps_per_chain: float = 0.0
    average_confidence: float = 0.0
    average_duration_ms: float = 0.0
    chains_by_agent: dict[str, int] = field(default_factory=dict)


@dataclass
class AnomalySummary:
    """Summary of anomaly detection statistics."""

    total_anomalies: int = 0
    anomalies_by_type: dict[str, int] = field(default_factory=dict)
    anomalies_by_severity: dict[str, int] = field(default_factory=dict)
    false_positive_rate: float = 0.0
    critical_anomalies: int = 0
    high_anomalies: int = 0


@dataclass
class ExplanationSummary:
    """Summary of explanation statistics."""

    total_explanations: int = 0
    explanations_by_type: dict[str, int] = field(default_factory=dict)
    average_contribution: float = 0.0
    top_features: list[str] = field(default_factory=list)


@dataclass
class DecisionSummary:
    """
    Summary of agent decision statistics.

    Sprint 1.7: Added for complete theme coverage.
    """

    total_decisions: int = 0
    decisions_by_agent: dict[str, int] = field(default_factory=dict)
    decisions_by_type: dict[str, int] = field(default_factory=dict)
    decisions_by_outcome: dict[str, int] = field(default_factory=dict)
    average_confidence: float = 0.0
    average_execution_time_ms: float = 0.0


@dataclass
class Sprint1MonitoringReport:
    """
    Unified monitoring report for Sprint 1 components.

    Sprint 1.5: Consolidated export for compliance and analysis.
    Sprint 1.7: Added decision_summary and context_count.
    """

    # Report metadata
    report_id: str
    generated_at: datetime
    report_period_start: datetime | None
    report_period_end: datetime | None

    # Component summaries
    reasoning_summary: ReasoningChainSummary
    anomaly_summary: AnomalySummary
    explanation_summary: ExplanationSummary

    # Sprint 1.7: Decision summary
    decision_summary: DecisionSummary = field(default_factory=DecisionSummary)
    context_count: int = 0  # UnifiedDecisionContext count

    # Raw data (optional, for detailed audit)
    include_raw_data: bool = False
    raw_reasoning_chains: list[dict[str, Any]] = field(default_factory=list)
    raw_anomalies: list[dict[str, Any]] = field(default_factory=list)
    raw_explanations: list[dict[str, Any]] = field(default_factory=list)
    raw_decisions: list[dict[str, Any]] = field(default_factory=list)  # Sprint 1.7

    # Health indicators
    overall_health_score: float = 1.0
    health_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "report_period_start": (self.report_period_start.isoformat() if self.report_period_start else None),
            "report_period_end": (self.report_period_end.isoformat() if self.report_period_end else None),
            "reasoning_summary": {
                "total_chains": self.reasoning_summary.total_chains,
                "completed_chains": self.reasoning_summary.completed_chains,
                "active_chains": self.reasoning_summary.active_chains,
                "failed_chains": self.reasoning_summary.failed_chains,
                "average_steps_per_chain": self.reasoning_summary.average_steps_per_chain,
                "average_confidence": self.reasoning_summary.average_confidence,
                "average_duration_ms": self.reasoning_summary.average_duration_ms,
                "chains_by_agent": self.reasoning_summary.chains_by_agent,
            },
            "anomaly_summary": {
                "total_anomalies": self.anomaly_summary.total_anomalies,
                "anomalies_by_type": self.anomaly_summary.anomalies_by_type,
                "anomalies_by_severity": self.anomaly_summary.anomalies_by_severity,
                "false_positive_rate": self.anomaly_summary.false_positive_rate,
                "critical_anomalies": self.anomaly_summary.critical_anomalies,
                "high_anomalies": self.anomaly_summary.high_anomalies,
            },
            "explanation_summary": {
                "total_explanations": self.explanation_summary.total_explanations,
                "explanations_by_type": self.explanation_summary.explanations_by_type,
                "average_contribution": self.explanation_summary.average_contribution,
                "top_features": self.explanation_summary.top_features,
            },
            "decision_summary": {
                "total_decisions": self.decision_summary.total_decisions,
                "decisions_by_agent": self.decision_summary.decisions_by_agent,
                "decisions_by_type": self.decision_summary.decisions_by_type,
                "decisions_by_outcome": self.decision_summary.decisions_by_outcome,
                "average_confidence": self.decision_summary.average_confidence,
                "average_execution_time_ms": self.decision_summary.average_execution_time_ms,
            },
            "context_count": self.context_count,
            "overall_health_score": self.overall_health_score,
            "health_warnings": self.health_warnings,
            "include_raw_data": self.include_raw_data,
            "raw_reasoning_chains": self.raw_reasoning_chains if self.include_raw_data else [],
            "raw_anomalies": self.raw_anomalies if self.include_raw_data else [],
            "raw_explanations": self.raw_explanations if self.include_raw_data else [],
            "raw_decisions": self.raw_decisions if self.include_raw_data else [],
        }


def generate_sprint1_report(
    reasoning_logger: ReasoningLogger | None = None,
    anomaly_detector: AnomalyDetector | None = None,
    explanation_logger: Any | None = None,
    decision_logger: Any | None = None,
    context_manager: Any | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    include_raw_data: bool = False,
) -> Sprint1MonitoringReport:
    """
    Generate unified Sprint 1 monitoring report.

    Sprint 1.5: Consolidated export combining all monitoring data.
    Sprint 1.7: Added decision_logger and context_manager support.

    Args:
        reasoning_logger: Optional ReasoningLogger instance
        anomaly_detector: Optional AnomalyDetector instance
        explanation_logger: Optional ExplanationLogger instance
        decision_logger: Optional DecisionLogger instance (Sprint 1.7)
        context_manager: Optional DecisionContextManager instance (Sprint 1.7)
        start_time: Filter start time
        end_time: Filter end time
        include_raw_data: Include raw data in report

    Returns:
        Sprint1MonitoringReport with aggregated statistics
    """
    import hashlib

    # Generate report ID
    timestamp = datetime.utcnow()
    content = f"sprint1_report:{timestamp.isoformat()}"
    report_id = hashlib.sha256(content.encode()).hexdigest()[:16]

    # Sprint 1.7: Collect decision summary
    decision_summary = DecisionSummary()
    raw_decisions: list[dict[str, Any]] = []
    context_count = 0

    if decision_logger:
        try:
            stats = decision_logger.get_statistics()
            decisions = decision_logger.get_decisions(limit=100)

            # Calculate by-agent distribution
            by_agent: dict[str, int] = {}
            by_type: dict[str, int] = {}
            by_outcome: dict[str, int] = {}
            total_conf = 0.0
            total_time = 0.0

            for d in decisions:
                by_agent[d.agent_name] = by_agent.get(d.agent_name, 0) + 1
                by_type[d.decision_type.value] = by_type.get(d.decision_type.value, 0) + 1
                by_outcome[d.outcome.value] = by_outcome.get(d.outcome.value, 0) + 1
                total_conf += d.confidence
                total_time += d.execution_time_ms

            decision_summary = DecisionSummary(
                total_decisions=len(decisions),
                decisions_by_agent=by_agent,
                decisions_by_type=by_type,
                decisions_by_outcome=by_outcome,
                average_confidence=total_conf / len(decisions) if decisions else 0.0,
                average_execution_time_ms=total_time / len(decisions) if decisions else 0.0,
            )

            if include_raw_data:
                raw_decisions = [d.to_dict() for d in decisions]
        except Exception:
            pass  # DecisionLogger may not have get_statistics

    if context_manager:
        try:
            stats = context_manager.get_statistics()
            context_count = stats.get("total_contexts", 0)
        except Exception:
            pass

    # Collect reasoning summary
    reasoning_summary = ReasoningChainSummary()
    raw_chains: list[dict[str, Any]] = []

    if reasoning_logger:
        stats = reasoning_logger.get_statistics()
        reasoning_summary = ReasoningChainSummary(
            total_chains=stats.get("total_chains", 0),
            completed_chains=stats.get("completed_chains", 0),
            active_chains=stats.get("active_chains", 0),
            failed_chains=stats.get("failed_chains", 0),
            average_steps_per_chain=stats.get("average_steps_per_chain", 0.0),
            average_confidence=stats.get("average_confidence", 0.0),
            average_duration_ms=stats.get("average_duration_ms", 0.0),
            chains_by_agent=stats.get("chains_by_agent", {}),
        )

        if include_raw_data:
            # Get all chains (would need to iterate through storage)
            pass  # Raw data collection would go here

    # Collect anomaly summary
    anomaly_summary = AnomalySummary()
    raw_anomalies: list[dict[str, Any]] = []

    if anomaly_detector:
        stats = anomaly_detector.get_statistics()
        by_severity = stats.get("anomalies_by_severity", {})

        anomaly_summary = AnomalySummary(
            total_anomalies=stats.get("anomalies_detected", 0),
            anomalies_by_type=stats.get("anomalies_by_type", {}),
            anomalies_by_severity=by_severity,
            false_positive_rate=stats.get("false_positive_rate_estimate", 0.0),
            critical_anomalies=by_severity.get("critical", 0),
            high_anomalies=by_severity.get("high", 0),
        )

        if include_raw_data:
            history = anomaly_detector.get_anomaly_history(limit=100)
            raw_anomalies = [a.to_dict() for a in history]

    # Collect explanation summary
    explanation_summary = ExplanationSummary()
    raw_explanations: list[dict[str, Any]] = []

    if explanation_logger:
        try:
            stats = explanation_logger.get_statistics()
            explanation_summary = ExplanationSummary(
                total_explanations=stats.get("total_explanations", 0),
                explanations_by_type=stats.get("explanations_by_type", {}),
                average_contribution=stats.get("average_contribution", 0.0),
                top_features=stats.get("top_features", []),
            )
        except Exception:
            pass  # ExplanationLogger may not have get_statistics

    # Calculate health score
    health_score = 1.0
    warnings: list[str] = []

    # Check for high critical anomaly rate
    if anomaly_summary.critical_anomalies > 5:
        health_score -= 0.2
        warnings.append(f"High critical anomaly count: {anomaly_summary.critical_anomalies}")

    # Check for high false positive rate
    if anomaly_summary.false_positive_rate > 0.1:
        health_score -= 0.1
        warnings.append(f"High false positive rate: {anomaly_summary.false_positive_rate:.1%}")

    # Check for failed reasoning chains
    if reasoning_summary.failed_chains > reasoning_summary.completed_chains * 0.1:
        health_score -= 0.1
        warnings.append("High reasoning chain failure rate")

    # Ensure health score is in [0, 1]
    health_score = max(0.0, min(1.0, health_score))

    return Sprint1MonitoringReport(
        report_id=report_id,
        generated_at=timestamp,
        report_period_start=start_time,
        report_period_end=end_time,
        reasoning_summary=reasoning_summary,
        anomaly_summary=anomaly_summary,
        explanation_summary=explanation_summary,
        decision_summary=decision_summary,
        context_count=context_count,
        include_raw_data=include_raw_data,
        raw_reasoning_chains=raw_chains,
        raw_anomalies=raw_anomalies,
        raw_explanations=raw_explanations,
        raw_decisions=raw_decisions,
        overall_health_score=health_score,
        health_warnings=warnings,
    )


def export_sprint1_report(
    report: Sprint1MonitoringReport,
    filepath: str,
) -> bool:
    """
    Export Sprint 1 monitoring report to JSON file.

    Args:
        report: The report to export
        filepath: Output file path

    Returns:
        True if exported successfully
    """
    try:
        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        return True
    except Exception:
        return False


def generate_sprint1_text_report(report: Sprint1MonitoringReport) -> str:
    """
    Generate human-readable text report.

    Args:
        report: The monitoring report

    Returns:
        Formatted text report
    """
    lines = [
        "=" * 60,
        "SPRINT 1 MONITORING REPORT",
        "=" * 60,
        "",
        f"Report ID: {report.report_id}",
        f"Generated: {report.generated_at.isoformat()}",
        f"Period: {report.report_period_start or 'N/A'} to {report.report_period_end or 'N/A'}",
        "",
        "-" * 40,
        "REASONING CHAINS",
        "-" * 40,
        f"Total Chains: {report.reasoning_summary.total_chains}",
        f"Completed: {report.reasoning_summary.completed_chains}",
        f"Active: {report.reasoning_summary.active_chains}",
        f"Failed: {report.reasoning_summary.failed_chains}",
        f"Avg Steps: {report.reasoning_summary.average_steps_per_chain:.2f}",
        f"Avg Confidence: {report.reasoning_summary.average_confidence:.2%}",
        f"Avg Duration: {report.reasoning_summary.average_duration_ms:.2f} ms",
        "",
        "-" * 40,
        "ANOMALY DETECTION",
        "-" * 40,
        f"Total Anomalies: {report.anomaly_summary.total_anomalies}",
        f"Critical: {report.anomaly_summary.critical_anomalies}",
        f"High: {report.anomaly_summary.high_anomalies}",
        f"False Positive Rate: {report.anomaly_summary.false_positive_rate:.1%}",
        "",
        "By Type:",
    ]

    for atype, count in report.anomaly_summary.anomalies_by_type.items():
        lines.append(f"  - {atype}: {count}")

    lines.extend(
        [
            "",
            "-" * 40,
            "EXPLANATIONS",
            "-" * 40,
            f"Total Explanations: {report.explanation_summary.total_explanations}",
            f"Avg Contribution: {report.explanation_summary.average_contribution:.4f}",
            "",
            "-" * 40,
            "AGENT DECISIONS (Sprint 1.7)",
            "-" * 40,
            f"Total Decisions: {report.decision_summary.total_decisions}",
            f"Avg Confidence: {report.decision_summary.average_confidence:.2%}",
            f"Avg Execution Time: {report.decision_summary.average_execution_time_ms:.2f} ms",
            f"Decision Contexts: {report.context_count}",
            "",
        ]
    )

    if report.decision_summary.decisions_by_agent:
        lines.append("By Agent:")
        for agent, count in report.decision_summary.decisions_by_agent.items():
            lines.append(f"  - {agent}: {count}")
        lines.append("")

    lines.extend(
        [
            "-" * 40,
            "HEALTH STATUS",
            "-" * 40,
            f"Overall Score: {report.overall_health_score:.0%}",
            "",
        ]
    )

    if report.health_warnings:
        lines.append("Warnings:")
        for warning in report.health_warnings:
            lines.append(f"  ! {warning}")
    else:
        lines.append("No warnings - all systems healthy")

    lines.extend(
        [
            "",
            "=" * 60,
        ]
    )

    return "\n".join(lines)
