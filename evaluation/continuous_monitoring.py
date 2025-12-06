"""
Continuous Monitoring System for Live Trading Evaluation.

Monitors live trading performance against evaluation baselines, detects drift,
and triggers alerts when performance degrades.

Features:
- Real-time performance tracking
- Drift detection (performance degradation from backtest)
- PSI-based drift detection (2025 research) - NEW
- Strategy health tracking with half-life calculation - NEW
- Automated alerting (email, Slack, webhook)
- Historical comparison and trending
- Dashboard metrics export (Prometheus/Grafana compatible)

References:
- https://onereach.ai/blog/agentic-ai-orchestration-enterprise-workflow-automation/
- https://www.kubiya.ai/blog/ai-orchestration-tools
- https://labelyourdata.com/articles/machine-learning/data-drift (Published: 2025)
- https://orq.ai/blog/model-vs-data-drift (Published: 2025)

Version: 2.0 (December 2025) - Added PSI drift detection and half-life tracking
"""

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# PSI drift detection (NEW - December 2025)
from evaluation.psi_drift_detection import (
    STRATEGY_HALF_LIFE_MONTHS,
    DriftLevel,
    PSIResult,
    StrategyHealthMetrics,
    calculate_psi_for_metric,
    calculate_strategy_health,
)

# Sprint 1.5: Import anomaly types for integration
from models.anomaly_detector import AnomalyResult


@dataclass
class PerformanceSnapshot:
    """Single performance snapshot from live trading."""

    timestamp: datetime
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    daily_return: float
    trades_count: int


@dataclass
class DriftAlert:
    """Alert for performance drift detection."""

    alert_id: str
    timestamp: datetime
    metric_name: str
    current_value: float
    baseline_value: float
    degradation_pct: float
    severity: str  # info | warning | critical
    message: str
    action_required: str


@dataclass
class PSIDriftAlert:
    """
    PSI-based drift alert (2025 research).

    Uses Population Stability Index for more rigorous drift detection
    compared to simple percentage thresholds.
    """

    alert_id: str
    timestamp: datetime
    metric_name: str
    psi_score: float
    drift_level: DriftLevel
    bin_contributions: list[float]  # Which bins are drifting most
    interpretation: str
    action_required: str


@dataclass
class MonitoringConfig:
    """Configuration for continuous monitoring."""

    # Drift detection thresholds (percentage-based)
    sharpe_drift_threshold: float = 0.20  # 20% degradation triggers alert
    win_rate_drift_threshold: float = 0.10  # 10% degradation
    drawdown_threshold: float = 0.20  # Max drawdown 20%

    # PSI drift detection thresholds (2025 research)
    psi_none_threshold: float = 0.10  # Below this = no drift
    psi_moderate_threshold: float = 0.25  # Below this = moderate drift (investigate)
    # Above psi_moderate_threshold = significant drift (action required)

    # Strategy health monitoring (half-life tracking)
    enable_halflife_tracking: bool = True
    strategy_start_date: datetime | None = None  # When strategy was deployed
    halflife_warning_pct: float = 0.75  # Warn when 75% of half-life reached
    halflife_months: float = STRATEGY_HALF_LIFE_MONTHS  # 11 months (2025 research)

    # PSI-based monitoring
    enable_psi_detection: bool = True
    psi_check_interval_snapshots: int = 10  # Check PSI every N snapshots
    min_snapshots_for_psi: int = 20  # Need at least 20 points for meaningful PSI

    # Alert channels
    email_alerts: bool = True
    slack_webhook: str | None = None
    webhook_url: str | None = None

    # Monitoring intervals
    snapshot_interval_minutes: int = 60  # Take snapshot every hour
    alert_cooldown_minutes: int = 180  # Don't spam alerts (3 hours cooldown)


class ContinuousMonitor:
    """
    Monitors live trading performance and detects drift from evaluation baselines.

    Compares real-time trading metrics against backtest/evaluation baselines
    to detect when the algorithm's performance degrades in live conditions.

    Enhanced with PSI drift detection (2025 research):
    - PSI < 0.10: No significant drift
    - 0.10 <= PSI < 0.25: Moderate drift (investigate)
    - PSI >= 0.25: Significant drift (action required)
    """

    def __init__(
        self,
        baseline_metrics: dict[str, float],
        config: MonitoringConfig | None = None,
        storage_path: Path | None = None,
        baseline_distributions: dict[str, list[float]] | None = None,
    ):
        """
        Initialize continuous monitor.

        Args:
            baseline_metrics: Baseline metrics from evaluation (backtest/walk-forward)
            config: Monitoring configuration
            storage_path: Path to store snapshots and alerts
            baseline_distributions: Historical distributions for PSI calculation (NEW)
        """
        self.baseline_metrics = baseline_metrics
        self.config = config or MonitoringConfig()
        self.storage_path = storage_path or Path("evaluation/monitoring")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Standard monitoring
        self.snapshots: list[PerformanceSnapshot] = []
        self.alerts: list[DriftAlert] = []
        self.last_alert_time: dict[str, datetime] = {}

        # PSI drift detection (NEW - December 2025)
        self.baseline_distributions = baseline_distributions or {}
        self.psi_alerts: list[PSIDriftAlert] = []
        self.latest_psi_results: dict[str, PSIResult] = {}
        self.latest_strategy_health: StrategyHealthMetrics | None = None

        # Track metric history for PSI calculation
        self._metric_history: dict[str, list[float]] = {
            "sharpe_ratio": [],
            "win_rate": [],
            "profit_factor": [],
            "max_drawdown": [],
            "daily_return": [],
        }

        # Sprint 1.5: Anomaly tracking (AnomalyDetector integration)
        self.anomaly_records: list[AnomalyResult] = []
        self._anomaly_counts_by_severity: dict[str, int] = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0,
        }
        self._anomaly_counts_by_type: dict[str, int] = {}

    def record_snapshot(self, snapshot: PerformanceSnapshot):
        """Record performance snapshot from live trading."""
        self.snapshots.append(snapshot)

        # Track metric history for PSI calculation
        self._metric_history["sharpe_ratio"].append(snapshot.sharpe_ratio)
        self._metric_history["win_rate"].append(snapshot.win_rate)
        self._metric_history["profit_factor"].append(snapshot.profit_factor)
        self._metric_history["max_drawdown"].append(snapshot.max_drawdown)
        self._metric_history["daily_return"].append(snapshot.daily_return)

        # Save to disk
        self._save_snapshot(snapshot)

        # Check for drift (percentage-based)
        self._check_drift(snapshot)

        # Check PSI-based drift if enabled
        if self.config.enable_psi_detection:
            self._check_psi_drift(snapshot)

        # Check strategy half-life if enabled
        if self.config.enable_halflife_tracking and self.config.strategy_start_date:
            self._check_strategy_health(snapshot)

    # =========================================================================
    # Sprint 1.5: Anomaly Recording Methods (AnomalyDetector Integration)
    # =========================================================================

    def record_anomaly(self, anomaly: AnomalyResult) -> None:
        """
        Record anomaly from AnomalyDetector.

        Sprint 1.5: Bridge between AnomalyDetector and ContinuousMonitor.
        Enables unified monitoring of both performance drift and market anomalies.

        Args:
            anomaly: AnomalyResult from AnomalyDetector.detect()
        """
        if not anomaly.is_anomaly:
            return  # Only record actual anomalies

        self.anomaly_records.append(anomaly)

        # Update severity counts
        severity_key = anomaly.severity.value.lower()
        if severity_key in self._anomaly_counts_by_severity:
            self._anomaly_counts_by_severity[severity_key] += 1

        # Update type counts
        type_key = anomaly.anomaly_type.value
        self._anomaly_counts_by_type[type_key] = self._anomaly_counts_by_type.get(type_key, 0) + 1

        # Save anomaly record
        self._save_anomaly_record(anomaly)

    def _save_anomaly_record(self, anomaly: AnomalyResult) -> None:
        """Save anomaly record to disk."""
        anomaly_file = self.storage_path / "anomalies.jsonl"

        anomaly_dict = {
            "timestamp": anomaly.timestamp.isoformat(),
            "is_anomaly": anomaly.is_anomaly,
            "anomaly_type": anomaly.anomaly_type.value,
            "severity": anomaly.severity.value,
            "score": anomaly.score,
            "description": anomaly.description,
            "recommended_action": anomaly.recommended_action,
            "feature_values": anomaly.feature_values,
        }

        with open(anomaly_file, "a") as f:
            f.write(json.dumps(anomaly_dict) + "\n")

    def get_anomaly_summary(self) -> dict[str, Any]:
        """
        Get summary of recorded anomalies.

        Sprint 1.5: Unified view of anomaly detection results.

        Returns:
            Dict with anomaly statistics
        """
        if not self.anomaly_records:
            return {"status": "No anomalies recorded"}

        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_anomalies = [a for a in self.anomaly_records if a.timestamp >= recent_cutoff]

        return {
            "total_anomalies": len(self.anomaly_records),
            "anomalies_last_24h": len(recent_anomalies),
            "by_severity": self._anomaly_counts_by_severity.copy(),
            "by_type": self._anomaly_counts_by_type.copy(),
            "critical_count": self._anomaly_counts_by_severity.get("critical", 0),
            "high_count": self._anomaly_counts_by_severity.get("high", 0),
            "latest_anomaly": (self.anomaly_records[-1].to_dict() if self.anomaly_records else None),
        }

    # =========================================================================
    # End Sprint 1.5 Anomaly Methods
    # =========================================================================

    def _check_drift(self, snapshot: PerformanceSnapshot):
        """Check for performance drift and trigger alerts."""
        checks = [
            (
                "sharpe_ratio",
                snapshot.sharpe_ratio,
                self.baseline_metrics.get("sharpe_ratio", 0),
                self.config.sharpe_drift_threshold,
            ),
            (
                "win_rate",
                snapshot.win_rate,
                self.baseline_metrics.get("win_rate", 0),
                self.config.win_rate_drift_threshold,
            ),
            (
                "profit_factor",
                snapshot.profit_factor,
                self.baseline_metrics.get("profit_factor", 1.0),
                0.30,  # 30% degradation for profit factor
            ),
        ]

        for metric_name, current_value, baseline_value, threshold in checks:
            if baseline_value > 0:
                degradation = (baseline_value - current_value) / baseline_value

                if degradation > threshold:
                    self._trigger_alert(
                        metric_name=metric_name,
                        current_value=current_value,
                        baseline_value=baseline_value,
                        degradation_pct=degradation,
                        snapshot_time=snapshot.timestamp,
                    )

        # Check max drawdown threshold
        if snapshot.max_drawdown > self.config.drawdown_threshold:
            self._trigger_alert(
                metric_name="max_drawdown",
                current_value=snapshot.max_drawdown,
                baseline_value=self.config.drawdown_threshold,
                degradation_pct=(snapshot.max_drawdown - self.config.drawdown_threshold)
                / self.config.drawdown_threshold,
                snapshot_time=snapshot.timestamp,
            )

    def _trigger_alert(
        self,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        degradation_pct: float,
        snapshot_time: datetime,
    ):
        """Trigger drift alert with cooldown."""
        # Check cooldown
        last_alert = self.last_alert_time.get(metric_name)
        if last_alert:
            time_since_last = (snapshot_time - last_alert).total_seconds() / 60
            if time_since_last < self.config.alert_cooldown_minutes:
                return  # Skip alert (cooldown active)

        # Determine severity
        if degradation_pct > 0.50:
            severity = "critical"
        elif degradation_pct > 0.30:
            severity = "warning"
        else:
            severity = "info"

        # Create alert
        alert = DriftAlert(
            alert_id=f"{metric_name}_{snapshot_time.strftime('%Y%m%d_%H%M%S')}",
            timestamp=snapshot_time,
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline_value,
            degradation_pct=degradation_pct,
            severity=severity,
            message=f"{metric_name} degraded {degradation_pct:.1%} from baseline "
            f"(current: {current_value:.2f}, baseline: {baseline_value:.2f})",
            action_required=self._get_action_recommendation(metric_name, degradation_pct),
        )

        self.alerts.append(alert)
        self.last_alert_time[metric_name] = snapshot_time

        # Send notifications
        self._send_alert(alert)

        # Save alert
        self._save_alert(alert)

    def _get_action_recommendation(self, metric_name: str, degradation_pct: float) -> str:
        """Get recommended action based on drift."""
        if degradation_pct > 0.50:
            return "CRITICAL: Consider halting live trading and re-evaluating algorithm"
        elif degradation_pct > 0.30:
            return "WARNING: Review recent trades and market conditions"
        else:
            return "INFO: Monitor closely for continued drift"

    def _send_alert(self, alert: DriftAlert):
        """Send alert notifications."""
        print(f"\n{'=' * 80}")
        print(f"🚨 DRIFT ALERT: {alert.severity.upper()}")
        print(f"{'=' * 80}")
        print(f"Metric: {alert.metric_name}")
        print(f"Current: {alert.current_value:.2f}")
        print(f"Baseline: {alert.baseline_value:.2f}")
        print(f"Degradation: {alert.degradation_pct:.1%}")
        print(f"Message: {alert.message}")
        print(f"Action: {alert.action_required}")
        print(f"{'=' * 80}\n")

        # TODO: Implement email/Slack/webhook notifications
        # if self.config.email_alerts:
        #     send_email_alert(alert)
        # if self.config.slack_webhook:
        #     send_slack_alert(alert, self.config.slack_webhook)
        # if self.config.webhook_url:
        #     send_webhook_alert(alert, self.config.webhook_url)

    def _save_snapshot(self, snapshot: PerformanceSnapshot):
        """Save snapshot to disk."""
        snapshot_file = self.storage_path / "snapshots" / f"{snapshot.timestamp.strftime('%Y%m%d')}.jsonl"
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)

        snapshot_dict = {
            "timestamp": snapshot.timestamp.isoformat(),
            "sharpe_ratio": snapshot.sharpe_ratio,
            "sortino_ratio": snapshot.sortino_ratio,
            "win_rate": snapshot.win_rate,
            "profit_factor": snapshot.profit_factor,
            "max_drawdown": snapshot.max_drawdown,
            "daily_return": snapshot.daily_return,
            "trades_count": snapshot.trades_count,
        }

        with open(snapshot_file, "a") as f:
            f.write(json.dumps(snapshot_dict) + "\n")

    def _save_alert(self, alert: DriftAlert):
        """Save alert to disk."""
        alert_file = self.storage_path / "alerts.jsonl"

        alert_dict = {
            "alert_id": alert.alert_id,
            "timestamp": alert.timestamp.isoformat(),
            "metric_name": alert.metric_name,
            "current_value": alert.current_value,
            "baseline_value": alert.baseline_value,
            "degradation_pct": alert.degradation_pct,
            "severity": alert.severity,
            "message": alert.message,
            "action_required": alert.action_required,
        }

        with open(alert_file, "a") as f:
            f.write(json.dumps(alert_dict) + "\n")

    # =========================================================================
    # PSI Drift Detection Methods (NEW - December 2025)
    # =========================================================================

    def _check_psi_drift(self, snapshot: PerformanceSnapshot):
        """
        Check for distribution shift using Population Stability Index.

        PSI Thresholds (2025 research):
        - PSI < 0.10: No significant shift
        - 0.10 <= PSI < 0.25: Moderate shift (investigate)
        - PSI >= 0.25: Significant shift (action required)
        """
        # Only check PSI at intervals to avoid excessive computation
        if len(self.snapshots) % self.config.psi_check_interval_snapshots != 0:
            return

        # Need minimum snapshots for meaningful PSI
        if len(self.snapshots) < self.config.min_snapshots_for_psi:
            return

        # Need baseline distributions
        if not self.baseline_distributions:
            return

        # Check PSI for each metric with baseline distribution
        for metric_name, baseline_dist in self.baseline_distributions.items():
            if metric_name not in self._metric_history:
                continue

            current_dist = self._metric_history[metric_name]
            if len(current_dist) < self.config.min_snapshots_for_psi:
                continue

            # Calculate PSI
            psi_result = calculate_psi_for_metric(
                metric_name=metric_name,
                expected_values=baseline_dist,
                actual_values=current_dist,
            )

            # Store latest result
            self.latest_psi_results[metric_name] = psi_result

            # Trigger alert if drift detected
            if psi_result.drift_level != DriftLevel.NONE:
                self._trigger_psi_alert(psi_result, snapshot.timestamp)

    def _trigger_psi_alert(self, psi_result: PSIResult, snapshot_time: datetime):
        """Trigger PSI-based drift alert with cooldown."""
        metric_name = f"psi_{psi_result.metric_name}"

        # Check cooldown
        last_alert = self.last_alert_time.get(metric_name)
        if last_alert:
            time_since_last = (snapshot_time - last_alert).total_seconds() / 60
            if time_since_last < self.config.alert_cooldown_minutes:
                return  # Skip alert (cooldown active)

        # Get action recommendation
        if psi_result.drift_level == DriftLevel.SIGNIFICANT:
            action = "CRITICAL: Significant distribution shift. Retrain/recalibrate strategy immediately."
        elif psi_result.drift_level == DriftLevel.MODERATE:
            action = "WARNING: Moderate distribution shift. Monitor closely and plan recalibration."
        else:
            action = "INFO: Minor shift detected. Continue monitoring."

        # Create PSI alert
        alert = PSIDriftAlert(
            alert_id=f"{metric_name}_{snapshot_time.strftime('%Y%m%d_%H%M%S')}",
            timestamp=snapshot_time,
            metric_name=psi_result.metric_name,
            psi_score=psi_result.psi_score,
            drift_level=psi_result.drift_level,
            bin_contributions=psi_result.bin_contributions,
            interpretation=psi_result.interpretation,
            action_required=action,
        )

        self.psi_alerts.append(alert)
        self.last_alert_time[metric_name] = snapshot_time

        # Send notification
        self._send_psi_alert(alert)

        # Save alert
        self._save_psi_alert(alert)

    def _send_psi_alert(self, alert: PSIDriftAlert):
        """Send PSI drift alert notification."""
        severity_emoji = {
            DriftLevel.NONE: "✅",
            DriftLevel.MODERATE: "⚠️",
            DriftLevel.SIGNIFICANT: "🔴",
        }

        print(f"\n{'=' * 80}")
        print(f"📊 PSI DRIFT ALERT: {severity_emoji[alert.drift_level]} {alert.drift_level.value.upper()}")
        print(f"{'=' * 80}")
        print(f"Metric: {alert.metric_name}")
        print(f"PSI Score: {alert.psi_score:.3f}")
        print(f"Drift Level: {alert.drift_level.value}")
        print(f"Interpretation: {alert.interpretation}")
        print(f"Action: {alert.action_required}")
        print(f"{'=' * 80}\n")

    def _save_psi_alert(self, alert: PSIDriftAlert):
        """Save PSI alert to disk."""
        alert_file = self.storage_path / "psi_alerts.jsonl"

        alert_dict = {
            "alert_id": alert.alert_id,
            "timestamp": alert.timestamp.isoformat(),
            "metric_name": alert.metric_name,
            "psi_score": alert.psi_score,
            "drift_level": alert.drift_level.value,
            "bin_contributions": alert.bin_contributions,
            "interpretation": alert.interpretation,
            "action_required": alert.action_required,
        }

        with open(alert_file, "a") as f:
            f.write(json.dumps(alert_dict) + "\n")

    def _check_strategy_health(self, snapshot: PerformanceSnapshot):
        """
        Check strategy health including half-life tracking.

        2025 Research findings:
        - AI strategy half-life: 11 months (down from 18 months in 2020)
        - Models unchanged for 6+ months see 35% error rate increase
        """
        if not self.config.strategy_start_date:
            return

        # Calculate strategy age
        age_days = (snapshot.timestamp - self.config.strategy_start_date).days
        age_months = age_days / 30.0

        # Check if approaching half-life
        halflife_pct = age_months / self.config.halflife_months

        if halflife_pct >= 1.0:
            # Past half-life - critical
            self._trigger_halflife_alert(
                age_months=age_months,
                halflife_pct=halflife_pct,
                severity="critical",
                snapshot_time=snapshot.timestamp,
            )
        elif halflife_pct >= self.config.halflife_warning_pct:
            # Approaching half-life - warning
            self._trigger_halflife_alert(
                age_months=age_months,
                halflife_pct=halflife_pct,
                severity="warning",
                snapshot_time=snapshot.timestamp,
            )

        # Calculate full strategy health if we have distributions
        if self.baseline_distributions and len(self.snapshots) >= self.config.min_snapshots_for_psi:
            self.latest_strategy_health = calculate_strategy_health(
                baseline_metrics=self.baseline_distributions,
                current_metrics=self._metric_history,
                strategy_start_date=self.config.strategy_start_date,
            )

    def _trigger_halflife_alert(
        self,
        age_months: float,
        halflife_pct: float,
        severity: str,
        snapshot_time: datetime,
    ):
        """Trigger half-life based alert."""
        metric_name = "strategy_halflife"

        # Check cooldown (use longer cooldown for half-life alerts - 24 hours)
        last_alert = self.last_alert_time.get(metric_name)
        if last_alert:
            time_since_last = (snapshot_time - last_alert).total_seconds() / 3600
            if time_since_last < 24:  # 24 hour cooldown
                return

        if severity == "critical":
            message = (
                f"Strategy age ({age_months:.1f} months) exceeds half-life "
                f"({self.config.halflife_months} months). Expected ~50% decay in effectiveness."
            )
            action = "CRITICAL: Full strategy review and update recommended."
        else:
            days_remaining = int((self.config.halflife_months - age_months) * 30)
            message = (
                f"Strategy approaching half-life ({halflife_pct:.0%} of {self.config.halflife_months} months). "
                f"~{days_remaining} days until recommended review."
            )
            action = "WARNING: Plan strategy review and recalibration."

        # Create alert using standard drift alert
        alert = DriftAlert(
            alert_id=f"{metric_name}_{snapshot_time.strftime('%Y%m%d_%H%M%S')}",
            timestamp=snapshot_time,
            metric_name=metric_name,
            current_value=age_months,
            baseline_value=self.config.halflife_months,
            degradation_pct=halflife_pct,
            severity=severity,
            message=message,
            action_required=action,
        )

        self.alerts.append(alert)
        self.last_alert_time[metric_name] = snapshot_time
        self._send_alert(alert)
        self._save_alert(alert)

    def get_psi_summary(self) -> dict[str, Any]:
        """Get summary of PSI drift detection results."""
        if not self.latest_psi_results:
            return {"status": "No PSI data available yet"}

        return {
            "metrics_monitored": list(self.latest_psi_results.keys()),
            "psi_scores": {name: result.psi_score for name, result in self.latest_psi_results.items()},
            "drift_levels": {name: result.drift_level.value for name, result in self.latest_psi_results.items()},
            "metrics_with_drift": [
                name for name, result in self.latest_psi_results.items() if result.drift_level != DriftLevel.NONE
            ],
            "psi_alerts_count": len(self.psi_alerts),
        }

    def get_strategy_health_summary(self) -> dict[str, Any]:
        """
        Get summary of strategy health status.

        Sprint 1.5: Now includes anomaly statistics for unified monitoring.
        """
        if not self.latest_strategy_health:
            return {"status": "No strategy health data available yet"}

        health = self.latest_strategy_health

        # Sprint 1.5: Include anomaly counts in health summary
        anomaly_summary = self.get_anomaly_summary()

        return {
            "overall_psi": health.overall_psi,
            "overall_drift_level": health.overall_drift_level.value,
            "strategy_age_months": health.strategy_age_months,
            "estimated_decay_pct": health.estimated_decay_pct,
            "days_until_review": health.days_until_review,
            "action_required": health.action_required,
            "recommendations": health.recommendations,
            "metrics_with_drift": health.metrics_with_drift,
            # Sprint 1.5: Anomaly integration
            "anomaly_counts": {
                "total": anomaly_summary.get("total_anomalies", 0),
                "last_24h": anomaly_summary.get("anomalies_last_24h", 0),
                "critical": anomaly_summary.get("critical_count", 0),
                "high": anomaly_summary.get("high_count", 0),
            },
        }

    # =========================================================================
    # End PSI Drift Detection Methods
    # =========================================================================

    def get_recent_performance(self, hours: int = 24) -> dict[str, Any]:
        """Get performance summary for recent period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]

        if not recent_snapshots:
            return {"error": "No snapshots in recent period"}

        return {
            "period_hours": hours,
            "snapshot_count": len(recent_snapshots),
            "avg_sharpe": statistics.mean([s.sharpe_ratio for s in recent_snapshots]),
            "avg_win_rate": statistics.mean([s.win_rate for s in recent_snapshots]),
            "avg_profit_factor": statistics.mean([s.profit_factor for s in recent_snapshots]),
            "max_drawdown": max([s.max_drawdown for s in recent_snapshots]),
            "total_trades": sum([s.trades_count for s in recent_snapshots]),
        }

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format for Grafana."""
        if not self.snapshots:
            return ""

        latest = self.snapshots[-1]

        metrics = []
        metrics.append(f"trading_sharpe_ratio {latest.sharpe_ratio}")
        metrics.append(f"trading_sortino_ratio {latest.sortino_ratio}")
        metrics.append(f"trading_win_rate {latest.win_rate}")
        metrics.append(f"trading_profit_factor {latest.profit_factor}")
        metrics.append(f"trading_max_drawdown {latest.max_drawdown}")
        metrics.append(f"trading_daily_return {latest.daily_return}")
        metrics.append(f"trading_trades_count {latest.trades_count}")

        # Drift metrics (percentage-based)
        for metric_name, baseline_value in self.baseline_metrics.items():
            current_value = getattr(latest, metric_name, 0)
            if baseline_value > 0:
                drift = (baseline_value - current_value) / baseline_value
                metrics.append(f"trading_drift_{metric_name} {drift}")

        # PSI metrics (NEW - December 2025)
        for metric_name, psi_result in self.latest_psi_results.items():
            metrics.append(f"trading_psi_{metric_name} {psi_result.psi_score}")
            # Drift level as numeric: 0=none, 1=moderate, 2=significant
            drift_level_num = {
                DriftLevel.NONE: 0,
                DriftLevel.MODERATE: 1,
                DriftLevel.SIGNIFICANT: 2,
            }.get(psi_result.drift_level, 0)
            metrics.append(f"trading_psi_drift_level_{metric_name} {drift_level_num}")

        # Strategy health metrics (half-life tracking)
        if self.latest_strategy_health:
            health = self.latest_strategy_health
            metrics.append(f"trading_strategy_age_months {health.strategy_age_months}")
            metrics.append(f"trading_strategy_decay_pct {health.estimated_decay_pct}")
            metrics.append(f"trading_strategy_days_until_review {health.days_until_review}")
            metrics.append(f"trading_strategy_overall_psi {health.overall_psi}")
            # Action required as numeric: 0=no, 1=yes
            action_num = 1 if health.action_required else 0
            metrics.append(f"trading_strategy_action_required {action_num}")

        return "\n".join(metrics)


def create_monitoring_dashboard_config() -> dict[str, Any]:
    """
    Create Grafana dashboard configuration for monitoring.

    Returns JSON config that can be imported into Grafana.
    Includes PSI drift detection panels (NEW - December 2025).
    """
    return {
        "dashboard": {
            "title": "Trading Algorithm Live Monitoring",
            "panels": [
                # Performance Metrics Row
                {
                    "title": "Sharpe Ratio (Live vs Baseline)",
                    "type": "graph",
                    "targets": [
                        {"expr": "trading_sharpe_ratio", "legendFormat": "Live"},
                        {"expr": "2.1", "legendFormat": "Baseline"},
                    ],
                },
                {
                    "title": "Win Rate (Live vs Baseline)",
                    "type": "graph",
                    "targets": [
                        {"expr": "trading_win_rate", "legendFormat": "Live"},
                        {"expr": "0.65", "legendFormat": "Baseline"},
                    ],
                },
                {
                    "title": "Performance Drift (Percentage)",
                    "type": "graph",
                    "targets": [
                        {"expr": "trading_drift_sharpe_ratio", "legendFormat": "Sharpe Drift"},
                        {"expr": "trading_drift_win_rate", "legendFormat": "Win Rate Drift"},
                    ],
                },
                {
                    "title": "Max Drawdown",
                    "type": "graph",
                    "targets": [
                        {"expr": "trading_max_drawdown", "legendFormat": "Current DD"},
                        {"expr": "0.20", "legendFormat": "Threshold (20%)"},
                    ],
                },
                # PSI Drift Detection Row (NEW - December 2025)
                {
                    "title": "PSI Scores by Metric",
                    "type": "graph",
                    "description": "PSI < 0.10: No drift | 0.10-0.25: Moderate | >= 0.25: Significant",
                    "targets": [
                        {"expr": "trading_psi_sharpe_ratio", "legendFormat": "Sharpe PSI"},
                        {"expr": "trading_psi_win_rate", "legendFormat": "Win Rate PSI"},
                        {"expr": "trading_psi_profit_factor", "legendFormat": "Profit Factor PSI"},
                        {"expr": "0.10", "legendFormat": "No Drift Threshold"},
                        {"expr": "0.25", "legendFormat": "Significant Threshold"},
                    ],
                },
                {
                    "title": "Overall Strategy PSI",
                    "type": "gauge",
                    "description": "Combined PSI across all metrics",
                    "targets": [
                        {"expr": "trading_strategy_overall_psi", "legendFormat": "Overall PSI"},
                    ],
                    "thresholds": {
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 0.10, "color": "yellow"},
                            {"value": 0.25, "color": "red"},
                        ]
                    },
                },
                # Strategy Health Row (NEW - December 2025)
                {
                    "title": "Strategy Age (Months)",
                    "type": "stat",
                    "description": f"Half-life: {STRATEGY_HALF_LIFE_MONTHS} months (2025 research)",
                    "targets": [
                        {"expr": "trading_strategy_age_months", "legendFormat": "Age"},
                    ],
                    "thresholds": {
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": STRATEGY_HALF_LIFE_MONTHS * 0.75, "color": "yellow"},
                            {"value": STRATEGY_HALF_LIFE_MONTHS, "color": "red"},
                        ]
                    },
                },
                {
                    "title": "Estimated Strategy Decay (%)",
                    "type": "gauge",
                    "description": "Based on 11-month half-life",
                    "targets": [
                        {"expr": "trading_strategy_decay_pct", "legendFormat": "Decay"},
                    ],
                    "thresholds": {
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 30, "color": "yellow"},
                            {"value": 50, "color": "red"},
                        ]
                    },
                },
                {
                    "title": "Days Until Review",
                    "type": "stat",
                    "description": "Countdown to recommended strategy review",
                    "targets": [
                        {"expr": "trading_strategy_days_until_review", "legendFormat": "Days"},
                    ],
                },
                {
                    "title": "Action Required",
                    "type": "stat",
                    "description": "0 = No, 1 = Yes",
                    "targets": [
                        {"expr": "trading_strategy_action_required", "legendFormat": "Action"},
                    ],
                    "thresholds": {
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 1, "color": "red"},
                        ]
                    },
                },
            ],
        }
    }


# =============================================================================
# Market Period Helpers (STOCKBENCH 2025 Integration)
# =============================================================================


class MarketPeriod:
    """Market period classification based on STOCKBENCH research."""

    DOWNTURN = "downturn"  # Jan-Apr: Model rankings shift
    UPTURN = "upturn"  # May-Aug: Different performance patterns
    NEUTRAL = "neutral"  # Sideways market


def get_market_period(date: datetime) -> str:
    """
    Determine market period based on date.

    STOCKBENCH 2025 research findings:
    - Jan-Apr: Downturn periods see different model rankings
    - May-Aug: Upturn periods show different performance patterns

    Args:
        date: Date to classify

    Returns:
        Market period string: "downturn", "upturn", or "neutral"
    """
    month = date.month

    if 1 <= month <= 4:
        return MarketPeriod.DOWNTURN
    elif 5 <= month <= 8:
        return MarketPeriod.UPTURN
    else:
        return MarketPeriod.NEUTRAL


def get_period_specific_thresholds(market_period: str) -> dict[str, Any]:
    """
    Get period-specific performance thresholds.

    Different market periods require different evaluation standards
    based on STOCKBENCH research findings.

    Args:
        market_period: "downturn", "upturn", or "neutral"

    Returns:
        Dict with adjusted thresholds for the period
    """
    base_thresholds = {
        "sharpe_drift_threshold": 0.20,
        "win_rate_drift_threshold": 0.10,
        "drawdown_threshold": 0.20,
        "psi_moderate_threshold": 0.25,
    }

    if market_period == MarketPeriod.DOWNTURN:
        # More lenient during downturns
        return {
            **base_thresholds,
            "sharpe_drift_threshold": 0.30,  # Allow 30% degradation
            "drawdown_threshold": 0.25,  # Allow higher drawdown
            "expected_sharpe_range": (0.5, 1.5),
            "note": "Downturn period: More lenient thresholds due to challenging conditions",
        }
    elif market_period == MarketPeriod.UPTURN:
        # Stricter during upturns
        return {
            **base_thresholds,
            "sharpe_drift_threshold": 0.15,  # Only 15% degradation allowed
            "win_rate_drift_threshold": 0.08,  # Tighter win rate
            "expected_sharpe_range": (1.5, 3.0),
            "note": "Upturn period: Stricter thresholds expected during favorable conditions",
        }
    else:
        return {
            **base_thresholds,
            "expected_sharpe_range": (1.0, 2.0),
            "note": "Neutral period: Standard thresholds apply",
        }


def create_period_aware_monitor(
    baseline_metrics: dict[str, float],
    strategy_start_date: datetime | None = None,
) -> ContinuousMonitor:
    """
    Create a ContinuousMonitor with period-aware thresholds.

    Automatically adjusts thresholds based on current market period.

    Args:
        baseline_metrics: Baseline metrics from evaluation
        strategy_start_date: When strategy was deployed

    Returns:
        ContinuousMonitor configured for current market period
    """
    current_period = get_market_period(datetime.now())
    thresholds = get_period_specific_thresholds(current_period)

    config = MonitoringConfig(
        sharpe_drift_threshold=thresholds["sharpe_drift_threshold"],
        win_rate_drift_threshold=thresholds["win_rate_drift_threshold"],
        drawdown_threshold=thresholds["drawdown_threshold"],
        psi_moderate_threshold=thresholds.get("psi_moderate_threshold", 0.25),
        enable_psi_detection=True,
        enable_halflife_tracking=True,
        strategy_start_date=strategy_start_date,
    )

    return ContinuousMonitor(
        baseline_metrics=baseline_metrics,
        config=config,
    )


def generate_period_report(monitor: ContinuousMonitor) -> str:
    """
    Generate monitoring report with market period context.

    Args:
        monitor: ContinuousMonitor instance

    Returns:
        Formatted markdown report with period awareness
    """
    current_period = get_market_period(datetime.now())
    thresholds = get_period_specific_thresholds(current_period)

    report = []
    report.append("# Continuous Monitoring Report (Period-Aware)\n")
    report.append(f"**Report Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Market Period**: {current_period.upper()}")
    report.append(f"**Period Note**: {thresholds.get('note', 'N/A')}\n")

    # Performance summary
    perf = monitor.get_recent_performance(hours=24)
    if "error" not in perf:
        report.append("## 24-Hour Performance Summary\n")
        report.append(f"- Snapshots: {perf['snapshot_count']}")
        report.append(f"- Avg Sharpe: {perf['avg_sharpe']:.3f}")
        report.append(f"- Avg Win Rate: {perf['avg_win_rate']:.1%}")
        report.append(f"- Max Drawdown: {perf['max_drawdown']:.1%}")
        report.append(f"- Total Trades: {perf['total_trades']}\n")

    # PSI Summary
    psi_summary = monitor.get_psi_summary()
    if "status" not in psi_summary:
        report.append("## PSI Drift Detection\n")
        for metric, score in psi_summary.get("psi_scores", {}).items():
            level = psi_summary.get("drift_levels", {}).get(metric, "unknown")
            emoji = "✅" if level == "none" else ("⚠️" if level == "moderate" else "🔴")
            report.append(f"- {metric}: PSI={score:.3f} ({level}) {emoji}")
        report.append("")

    # Strategy Health
    health = monitor.get_strategy_health_summary()
    if "status" not in health:
        report.append("## Strategy Health\n")
        report.append(f"- Age: {health['strategy_age_months']:.1f} months")
        report.append(f"- Estimated Decay: {health['estimated_decay_pct']:.1f}%")
        report.append(f"- Days Until Review: {health['days_until_review']}")
        report.append(f"- Action Required: {'YES' if health['action_required'] else 'No'}")
        report.append("\n### Recommendations\n")
        for rec in health.get("recommendations", []):
            report.append(f"- {rec}")

    # Period-specific expectations
    report.append(f"\n## Period-Specific Expectations ({current_period.upper()})\n")
    expected_range = thresholds.get("expected_sharpe_range", (1.0, 2.0))
    report.append(f"- Expected Sharpe Range: {expected_range[0]:.1f} - {expected_range[1]:.1f}")
    report.append(f"- Sharpe Drift Threshold: {thresholds['sharpe_drift_threshold']:.1%}")
    report.append(f"- Drawdown Threshold: {thresholds['drawdown_threshold']:.1%}")

    return "\n".join(report)
