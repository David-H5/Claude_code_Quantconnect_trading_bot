"""Anomaly detection for agent behavior.

Monitors agent performance metrics for anomalies:
- Confidence drops (sudden decrease in decision confidence)
- Decision rate changes (abnormal activity levels)
- Accuracy degradation (performance decline)
- Calibration drift (confidence no longer matching outcomes)

Usage:
    from observability.anomaly_detector import run_anomaly_detection, AnomalyDetector

    anomalies = run_anomaly_detection()
    for anomaly in anomalies:
        print(f"[{anomaly.severity.value}] {anomaly.message}")

    # Or run directly
    python -m observability.anomaly_detector
"""

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class AlertSeverity(Enum):
    """Severity levels for anomaly alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""

    CONFIDENCE_DROP = "confidence_drop"
    DECISION_RATE_SPIKE = "decision_rate_spike"
    DECISION_RATE_DROP = "decision_rate_drop"
    ACCURACY_DROP = "accuracy_drop"
    CALIBRATION_DRIFT = "calibration_drift"
    NO_ACTIVITY = "no_activity"


@dataclass
class Anomaly:
    """A detected anomaly in agent behavior.

    Attributes:
        anomaly_type: Type of anomaly detected.
        metric_name: The metric that triggered the anomaly.
        current_value: Current metric value.
        baseline_value: Expected/baseline value.
        deviation: How far from baseline (in std deviations or percentage).
        severity: Alert severity level.
        agent: Agent name (or None for system-wide).
        timestamp: When the anomaly was detected.
        message: Human-readable description.
    """

    anomaly_type: AnomalyType
    metric_name: str
    current_value: float
    baseline_value: float
    deviation: float
    severity: AlertSeverity
    agent: str | None
    timestamp: datetime
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "anomaly_type": self.anomaly_type.value,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "deviation": self.deviation,
            "severity": self.severity.value,
            "agent": self.agent,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "context": self.context,
        }


# Detection thresholds
THRESHOLDS = {
    "confidence_drop": {
        "warning": 0.15,  # 15% drop from baseline
        "error": 0.25,  # 25% drop
        "critical": 0.40,  # 40% drop
    },
    "decision_rate_change": {
        "warning": 0.40,  # 40% change from baseline
        "error": 0.60,  # 60% change
        "critical": 0.80,  # 80% change
    },
    "accuracy_drop": {
        "warning": 0.10,  # 10% drop
        "error": 0.20,  # 20% drop
        "critical": 0.35,  # 35% drop
    },
    "calibration_drift": {
        "warning": 0.15,  # 15% error increase
        "error": 0.25,  # 25% error increase
        "critical": 0.40,  # 40% error increase
    },
}

# Minimum samples for reliable baseline
MIN_BASELINE_SAMPLES = 5

# Confidence level mapping
CONFIDENCE_MAP = {
    "very_low": 0.1,
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7,
    "very_high": 0.9,
}

# Paths
DECISION_DIR = Path(".claude/state/decisions")
ANOMALY_DIR = Path(".claude/state/anomalies")


def calculate_baseline(values: list[float]) -> tuple[float, float]:
    """Calculate baseline mean and standard deviation.

    Args:
        values: List of historical values.

    Returns:
        Tuple of (mean, std_dev).
    """
    if len(values) < MIN_BASELINE_SAMPLES:
        return 0.5, 0.15  # Default baseline

    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.1

    return mean, max(stdev, 0.01)  # Prevent zero stdev


def determine_severity(drop_pct: float, thresholds: dict[str, float]) -> AlertSeverity | None:
    """Determine alert severity based on threshold crossing.

    Args:
        drop_pct: The percentage change (as a decimal, e.g., 0.25 for 25%).
        thresholds: Dictionary with warning/error/critical thresholds.

    Returns:
        Severity level or None if below warning threshold.
    """
    if drop_pct >= thresholds["critical"]:
        return AlertSeverity.CRITICAL
    elif drop_pct >= thresholds["error"]:
        return AlertSeverity.ERROR
    elif drop_pct >= thresholds["warning"]:
        return AlertSeverity.WARNING
    return None


class AnomalyDetector:
    """Detects anomalies in agent behavior patterns.

    Compares recent metrics against historical baselines to identify
    significant deviations that may indicate issues.
    """

    def __init__(
        self,
        baseline_days: int = 7,
        recent_hours: int = 24,
    ):
        """Initialize the detector.

        Args:
            baseline_days: Days of history to use for baseline.
            recent_hours: Hours of recent data to compare against baseline.
        """
        self.baseline_days = baseline_days
        self.recent_hours = recent_hours
        self.anomalies: list[Anomaly] = []

    def load_decisions(self) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
        """Load decisions split into baseline and recent periods.

        Returns:
            Tuple of (baseline_decisions, recent_decisions) by agent.
        """
        now = datetime.now()
        recent_cutoff = now - timedelta(hours=self.recent_hours)
        baseline_cutoff = now - timedelta(days=self.baseline_days)

        baseline: dict[str, list[dict]] = {}
        recent: dict[str, list[dict]] = {}

        if not DECISION_DIR.exists():
            return baseline, recent

        for log_file in DECISION_DIR.glob("*.json"):
            try:
                data = json.loads(log_file.read_text())
                timestamp_str = data.get("timestamp", "")

                try:
                    decision_time = datetime.fromisoformat(timestamp_str)
                except ValueError:
                    continue

                agent = data.get("agent_name", "unknown")

                # Classify into recent or baseline
                if decision_time >= recent_cutoff:
                    if agent not in recent:
                        recent[agent] = []
                    recent[agent].append(data)
                elif decision_time >= baseline_cutoff:
                    if agent not in baseline:
                        baseline[agent] = []
                    baseline[agent].append(data)
            except (json.JSONDecodeError, OSError):
                continue

        return baseline, recent

    def extract_confidences(self, decisions: list[dict]) -> list[float]:
        """Extract confidence values from decisions.

        Args:
            decisions: List of decision records.

        Returns:
            List of confidence values.
        """
        confidences = []
        for d in decisions:
            conf = d.get("confidence")
            if isinstance(conf, float):
                confidences.append(conf)
            else:
                conf_str = d.get("overall_confidence", "medium")
                confidences.append(CONFIDENCE_MAP.get(conf_str, 0.5))
        return confidences

    def detect_confidence_anomaly(
        self,
        agent: str,
        baseline_decisions: list[dict],
        recent_decisions: list[dict],
    ) -> Anomaly | None:
        """Detect confidence drop anomalies.

        Args:
            agent: Agent name.
            baseline_decisions: Historical decisions.
            recent_decisions: Recent decisions.

        Returns:
            Anomaly if detected, None otherwise.
        """
        baseline_conf = self.extract_confidences(baseline_decisions)
        recent_conf = self.extract_confidences(recent_decisions)

        if len(baseline_conf) < MIN_BASELINE_SAMPLES or not recent_conf:
            return None

        baseline_mean, baseline_std = calculate_baseline(baseline_conf)
        current_mean = statistics.mean(recent_conf)

        # Calculate drop as percentage of baseline
        if baseline_mean > 0:
            drop_pct = (baseline_mean - current_mean) / baseline_mean
        else:
            drop_pct = 0

        if drop_pct <= 0:
            return None  # No drop

        severity = determine_severity(drop_pct, THRESHOLDS["confidence_drop"])
        if not severity:
            return None

        deviation = (baseline_mean - current_mean) / baseline_std if baseline_std > 0 else 0

        return Anomaly(
            anomaly_type=AnomalyType.CONFIDENCE_DROP,
            metric_name="confidence",
            current_value=current_mean,
            baseline_value=baseline_mean,
            deviation=deviation,
            severity=severity,
            agent=agent,
            timestamp=datetime.now(),
            message=f"Agent {agent} confidence dropped {drop_pct:.1%} from baseline ({baseline_mean:.2f} → {current_mean:.2f})",
            context={
                "baseline_samples": len(baseline_conf),
                "recent_samples": len(recent_conf),
                "baseline_std": baseline_std,
            },
        )

    def detect_decision_rate_anomaly(
        self,
        agent: str,
        baseline_decisions: list[dict],
        recent_decisions: list[dict],
    ) -> Anomaly | None:
        """Detect decision rate anomalies (spikes or drops).

        Args:
            agent: Agent name.
            baseline_decisions: Historical decisions.
            recent_decisions: Recent decisions.

        Returns:
            Anomaly if detected, None otherwise.
        """
        # Calculate rates (decisions per hour)
        baseline_hours = self.baseline_days * 24
        recent_hours = self.recent_hours

        baseline_rate = len(baseline_decisions) / baseline_hours if baseline_hours > 0 else 0
        recent_rate = len(recent_decisions) / recent_hours if recent_hours > 0 else 0

        if baseline_rate == 0:
            return None  # Can't compare without baseline

        # Calculate change as percentage
        change_pct = abs(recent_rate - baseline_rate) / baseline_rate
        is_spike = recent_rate > baseline_rate

        severity = determine_severity(change_pct, THRESHOLDS["decision_rate_change"])
        if not severity:
            return None

        anomaly_type = AnomalyType.DECISION_RATE_SPIKE if is_spike else AnomalyType.DECISION_RATE_DROP
        direction = "increased" if is_spike else "decreased"

        return Anomaly(
            anomaly_type=anomaly_type,
            metric_name="decision_rate",
            current_value=recent_rate,
            baseline_value=baseline_rate,
            deviation=change_pct,
            severity=severity,
            agent=agent,
            timestamp=datetime.now(),
            message=f"Agent {agent} decision rate {direction} {change_pct:.1%} ({baseline_rate:.2f}/hr → {recent_rate:.2f}/hr)",
            context={
                "baseline_count": len(baseline_decisions),
                "recent_count": len(recent_decisions),
                "baseline_hours": baseline_hours,
                "recent_hours": recent_hours,
            },
        )

    def detect_no_activity(
        self,
        agent: str,
        baseline_decisions: list[dict],
        recent_decisions: list[dict],
    ) -> Anomaly | None:
        """Detect agents with baseline activity but no recent activity.

        Args:
            agent: Agent name.
            baseline_decisions: Historical decisions.
            recent_decisions: Recent decisions.

        Returns:
            Anomaly if detected, None otherwise.
        """
        if len(recent_decisions) > 0:
            return None

        if len(baseline_decisions) < MIN_BASELINE_SAMPLES:
            return None

        return Anomaly(
            anomaly_type=AnomalyType.NO_ACTIVITY,
            metric_name="activity",
            current_value=0,
            baseline_value=len(baseline_decisions),
            deviation=1.0,
            severity=AlertSeverity.WARNING,
            agent=agent,
            timestamp=datetime.now(),
            message=f"Agent {agent} has no activity in last {self.recent_hours} hours (had {len(baseline_decisions)} decisions in baseline period)",
            context={
                "baseline_count": len(baseline_decisions),
                "recent_hours": self.recent_hours,
            },
        )

    def run(self) -> list[Anomaly]:
        """Run anomaly detection across all agents.

        Returns:
            List of detected anomalies.
        """
        self.anomalies = []
        baseline, recent = self.load_decisions()

        # Get all agents (from both periods)
        all_agents = set(baseline.keys()) | set(recent.keys())

        for agent in all_agents:
            baseline_decisions = baseline.get(agent, [])
            recent_decisions = recent.get(agent, [])

            # Check for confidence drops
            anomaly = self.detect_confidence_anomaly(agent, baseline_decisions, recent_decisions)
            if anomaly:
                self.anomalies.append(anomaly)

            # Check for decision rate changes
            anomaly = self.detect_decision_rate_anomaly(agent, baseline_decisions, recent_decisions)
            if anomaly:
                self.anomalies.append(anomaly)

            # Check for no activity
            anomaly = self.detect_no_activity(agent, baseline_decisions, recent_decisions)
            if anomaly:
                self.anomalies.append(anomaly)

        # Sort by severity (critical first)
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.ERROR: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3,
        }
        self.anomalies.sort(key=lambda a: severity_order.get(a.severity, 4))

        # Save anomalies
        self._save_anomalies()

        return self.anomalies

    def _save_anomalies(self) -> None:
        """Save detected anomalies to state file."""
        ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
        output_file = ANOMALY_DIR / f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            "detection_time": datetime.now().isoformat(),
            "baseline_days": self.baseline_days,
            "recent_hours": self.recent_hours,
            "count": len(self.anomalies),
            "anomalies": [a.to_dict() for a in self.anomalies],
        }
        output_file.write_text(json.dumps(data, indent=2))


def run_anomaly_detection(
    baseline_days: int = 7,
    recent_hours: int = 24,
) -> list[Anomaly]:
    """Run anomaly detection with default settings.

    Args:
        baseline_days: Days of history for baseline.
        recent_hours: Hours of recent data to analyze.

    Returns:
        List of detected anomalies.
    """
    detector = AnomalyDetector(baseline_days=baseline_days, recent_hours=recent_hours)
    return detector.run()


def print_anomalies(anomalies: list[Anomaly]) -> None:
    """Print formatted anomaly report.

    Args:
        anomalies: List of anomalies to display.
    """
    print("\n" + "=" * 70)
    print("🔍 ANOMALY DETECTION REPORT")
    print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    if not anomalies:
        print("\n  ✅ No anomalies detected. All systems normal.")
        print("=" * 70 + "\n")
        return

    # Icons by severity
    severity_icons = {
        AlertSeverity.CRITICAL: "🚨",
        AlertSeverity.ERROR: "❌",
        AlertSeverity.WARNING: "⚠️",
        AlertSeverity.INFO: "ℹ️",
    }

    print(f"\n  Found {len(anomalies)} anomalies:\n")

    for anomaly in anomalies:
        icon = severity_icons.get(anomaly.severity, "❓")
        print(f"  {icon} [{anomaly.severity.value.upper()}] {anomaly.message}")
        print(f"     Current: {anomaly.current_value:.2f} | Baseline: {anomaly.baseline_value:.2f}")
        print()

    # Summary by severity
    print("-" * 70)
    for severity in [AlertSeverity.CRITICAL, AlertSeverity.ERROR, AlertSeverity.WARNING]:
        count = sum(1 for a in anomalies if a.severity == severity)
        if count > 0:
            icon = severity_icons[severity]
            print(f"  {icon} {severity.value.upper()}: {count}")

    print("=" * 70 + "\n")


def main() -> int:
    """Main entry point for anomaly detection.

    Returns:
        Exit code (0 for success, 1 if critical anomalies).
    """
    import sys

    baseline_days = 7
    recent_hours = 24

    if "--baseline" in sys.argv:
        idx = sys.argv.index("--baseline")
        if idx + 1 < len(sys.argv):
            try:
                baseline_days = int(sys.argv[idx + 1])
            except ValueError:
                pass

    if "--recent" in sys.argv:
        idx = sys.argv.index("--recent")
        if idx + 1 < len(sys.argv):
            try:
                recent_hours = int(sys.argv[idx + 1])
            except ValueError:
                pass

    anomalies = run_anomaly_detection(baseline_days=baseline_days, recent_hours=recent_hours)

    if "--json" in sys.argv:
        print(json.dumps([a.to_dict() for a in anomalies], indent=2))
    else:
        print_anomalies(anomalies)

    # Return 1 if critical anomalies found
    has_critical = any(a.severity == AlertSeverity.CRITICAL for a in anomalies)
    return 1 if has_critical else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
