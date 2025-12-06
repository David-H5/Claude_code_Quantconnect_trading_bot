"""
Anomaly Alerting Bridge (UPGRADE-010 Sprint 1.7)

Connects AnomalyDetector to AlertingService and ContinuousMonitor,
creating a full Continuous→Alerting→Action pipeline.

Features:
- Automatic anomaly → alert conversion
- Severity-based action triggers
- ContinuousMonitor integration
- Action callbacks for trading decisions
- Configurable thresholds

Part of UPGRADE-010: Advanced AI Features
Phase: Sprint 1.7 (Theme Completion)

QuantConnect Compatible: Yes
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from models.anomaly_detector import AnomalyDetector, AnomalyResult, AnomalySeverity


logger = logging.getLogger(__name__)


class AlertingAction(Enum):
    """Actions to take based on anomaly severity."""

    CONTINUE = "continue"  # No action, continue trading
    MONITOR = "monitor"  # Log and continue, heightened awareness
    REDUCE_EXPOSURE = "reduce_exposure"  # Reduce position sizes
    HALT_NEW_ORDERS = "halt_new_orders"  # Stop opening new positions
    HALT_TRADING = "halt_trading"  # Full trading halt


@dataclass
class AlertingPipelineConfig:
    """Configuration for the alerting pipeline."""

    # Severity → Action mapping
    action_on_low: AlertingAction = AlertingAction.CONTINUE
    action_on_medium: AlertingAction = AlertingAction.MONITOR
    action_on_high: AlertingAction = AlertingAction.REDUCE_EXPOSURE
    action_on_critical: AlertingAction = AlertingAction.HALT_TRADING

    # Alert filtering
    min_severity_for_alert: AnomalySeverity = AnomalySeverity.MEDIUM
    min_severity_for_action: AnomalySeverity = AnomalySeverity.HIGH

    # Continuous monitor integration
    enable_continuous_monitor: bool = True

    # Alert service integration
    enable_alerting_service: bool = True

    # Action callback settings
    enable_action_callbacks: bool = True


class AnomalyAlertingBridge:
    """
    Bridge connecting AnomalyDetector to AlertingService and ContinuousMonitor.

    Sprint 1.7: Creates the full Continuous→Alerting→Action pipeline.

    Usage:
        from evaluation import create_alerting_pipeline

        # Create pipeline
        pipeline = create_alerting_pipeline(
            anomaly_detector=detector,
            alerting_service=alerting_service,
            continuous_monitor=monitor,
        )

        # Register action callback
        pipeline.on_action(AlertingAction.HALT_TRADING, circuit_breaker.trip)

        # Pipeline is now active - anomalies trigger alerts and actions
    """

    def __init__(
        self,
        anomaly_detector: AnomalyDetector | None = None,
        alerting_service: Any | None = None,
        continuous_monitor: Any | None = None,
        config: AlertingPipelineConfig | None = None,
    ):
        """
        Initialize the alerting bridge.

        Args:
            anomaly_detector: AnomalyDetector to observe
            alerting_service: AlertingService to send alerts to
            continuous_monitor: ContinuousMonitor to record anomalies
            config: Pipeline configuration
        """
        self.anomaly_detector = anomaly_detector
        self.alerting_service = alerting_service
        self.continuous_monitor = continuous_monitor
        self.config = config or AlertingPipelineConfig()

        # Action callbacks by action type
        self._action_callbacks: dict[AlertingAction, list[Callable[[], None]]] = {
            action: [] for action in AlertingAction
        }

        # Statistics
        self._alerts_sent: int = 0
        self._actions_triggered: int = 0
        self._anomalies_processed: int = 0
        self._last_action: AlertingAction | None = None
        self._last_action_time: datetime | None = None

        # Register as observer if detector provided
        if self.anomaly_detector:
            self.anomaly_detector.add_observer(self._on_anomaly)

    def _on_anomaly(self, result: AnomalyResult) -> None:
        """
        Handle anomaly from detector.

        This is the main entry point for the pipeline.
        """
        self._anomalies_processed += 1

        # 1. Record in continuous monitor
        if self.config.enable_continuous_monitor and self.continuous_monitor:
            try:
                self.continuous_monitor.record_anomaly(result)
            except Exception as e:
                logger.warning(f"Failed to record anomaly in monitor: {e}")

        # 2. Send alert if severity meets threshold
        if self.config.enable_alerting_service and self.alerting_service:
            if result.severity.value >= self.config.min_severity_for_alert.value:
                self._send_alert(result)

        # 3. Trigger action if severity meets threshold
        if self.config.enable_action_callbacks:
            if result.severity.value >= self.config.min_severity_for_action.value:
                action = self._get_action_for_severity(result.severity)
                self._trigger_action(action, result)

    def _send_alert(self, result: AnomalyResult) -> bool:
        """Send alert through alerting service."""
        if not self.alerting_service:
            return False

        try:
            # Map severity to alert severity
            from utils.alerting_service import AlertSeverity

            severity_map = {
                AnomalySeverity.LOW: AlertSeverity.INFO,
                AnomalySeverity.MEDIUM: AlertSeverity.WARNING,
                AnomalySeverity.HIGH: AlertSeverity.ERROR,
                AnomalySeverity.CRITICAL: AlertSeverity.CRITICAL,
            }
            alert_severity = severity_map.get(result.severity, AlertSeverity.WARNING)

            # Send via alerting service
            sent = self.alerting_service.send_anomaly_alert(
                anomaly_type=result.anomaly_type.value,
                message=result.description,
                severity=alert_severity,
                score=result.score,
                recommended_action=result.recommended_action,
                data=result.feature_values,
            )

            if sent:
                self._alerts_sent += 1
            return sent

        except Exception as e:
            logger.warning(f"Failed to send anomaly alert: {e}")
            return False

    def _get_action_for_severity(self, severity: AnomalySeverity) -> AlertingAction:
        """Get action for severity level."""
        action_map = {
            AnomalySeverity.LOW: self.config.action_on_low,
            AnomalySeverity.MEDIUM: self.config.action_on_medium,
            AnomalySeverity.HIGH: self.config.action_on_high,
            AnomalySeverity.CRITICAL: self.config.action_on_critical,
        }
        return action_map.get(severity, AlertingAction.MONITOR)

    def _trigger_action(self, action: AlertingAction, result: AnomalyResult) -> None:
        """Trigger action callbacks for the given action."""
        if action == AlertingAction.CONTINUE:
            return  # No action needed

        self._last_action = action
        self._last_action_time = datetime.now(timezone.utc)
        self._actions_triggered += 1

        # Call registered callbacks
        callbacks = self._action_callbacks.get(action, [])
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Action callback failed for {action.value}: {e}")

        logger.info(f"Triggered action {action.value} for anomaly: {result.description}")

    def on_action(
        self,
        action: AlertingAction,
        callback: Callable[[], None],
    ) -> None:
        """
        Register a callback for a specific action.

        Args:
            action: The action type to register for
            callback: Function to call when action is triggered

        Example:
            pipeline.on_action(AlertingAction.HALT_TRADING, circuit_breaker.trip)
            pipeline.on_action(AlertingAction.REDUCE_EXPOSURE, risk_manager.reduce_exposure)
        """
        self._action_callbacks[action].append(callback)

    def remove_action_callback(
        self,
        action: AlertingAction,
        callback: Callable[[], None],
    ) -> bool:
        """
        Remove a callback for a specific action.

        Args:
            action: The action type
            callback: The callback to remove

        Returns:
            True if callback was removed
        """
        callbacks = self._action_callbacks.get(action, [])
        if callback in callbacks:
            callbacks.remove(callback)
            return True
        return False

    def disconnect(self) -> None:
        """
        Disconnect from anomaly detector.

        Call this when disposing of the bridge.
        """
        if self.anomaly_detector:
            self.anomaly_detector.remove_observer(self._on_anomaly)

    def get_statistics(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "anomalies_processed": self._anomalies_processed,
            "alerts_sent": self._alerts_sent,
            "actions_triggered": self._actions_triggered,
            "last_action": self._last_action.value if self._last_action else None,
            "last_action_time": (self._last_action_time.isoformat() if self._last_action_time else None),
            "action_callbacks_registered": {
                action.value: len(callbacks) for action, callbacks in self._action_callbacks.items()
            },
            "config": {
                "min_severity_for_alert": self.config.min_severity_for_alert.value,
                "min_severity_for_action": self.config.min_severity_for_action.value,
                "alerting_enabled": self.config.enable_alerting_service,
                "monitoring_enabled": self.config.enable_continuous_monitor,
            },
        }


def create_alerting_pipeline(
    anomaly_detector: AnomalyDetector | None = None,
    alerting_service: Any | None = None,
    continuous_monitor: Any | None = None,
    config: AlertingPipelineConfig | None = None,
) -> AnomalyAlertingBridge:
    """
    Factory function to create an alerting pipeline.

    Sprint 1.7: Creates the full Continuous→Alerting→Action pipeline.

    Args:
        anomaly_detector: AnomalyDetector to observe
        alerting_service: AlertingService to send alerts to
        continuous_monitor: ContinuousMonitor to record anomalies
        config: Pipeline configuration

    Returns:
        Configured AnomalyAlertingBridge

    Example:
        # Create all components
        detector = create_anomaly_detector()
        alerting = create_alerting_service()
        monitor = ContinuousMonitor(baseline_metrics)

        # Create pipeline
        pipeline = create_alerting_pipeline(
            anomaly_detector=detector,
            alerting_service=alerting,
            continuous_monitor=monitor,
        )

        # Register action callbacks
        pipeline.on_action(AlertingAction.HALT_TRADING, lambda: breaker.trip())
        pipeline.on_action(AlertingAction.REDUCE_EXPOSURE, lambda: risk.reduce(0.5))
    """
    return AnomalyAlertingBridge(
        anomaly_detector=anomaly_detector,
        alerting_service=alerting_service,
        continuous_monitor=continuous_monitor,
        config=config,
    )


def create_full_sprint1_pipeline(
    anomaly_detector: AnomalyDetector | None = None,
    alerting_service: Any | None = None,
    continuous_monitor: Any | None = None,
    decision_context_manager: Any | None = None,
    reasoning_logger: Any | None = None,
    config: AlertingPipelineConfig | None = None,
) -> dict[str, Any]:
    """
    Create a fully integrated Sprint 1 monitoring pipeline.

    Sprint 1.7: Complete integration of all Sprint 1 components.

    Args:
        anomaly_detector: AnomalyDetector instance
        alerting_service: AlertingService instance
        continuous_monitor: ContinuousMonitor instance
        decision_context_manager: DecisionContextManager instance
        reasoning_logger: ReasoningLogger instance
        config: Pipeline configuration

    Returns:
        Dict with all pipeline components:
        - "bridge": AnomalyAlertingBridge
        - "anomaly_detector": AnomalyDetector
        - "alerting_service": AlertingService
        - "continuous_monitor": ContinuousMonitor
        - "decision_context_manager": DecisionContextManager
        - "reasoning_logger": ReasoningLogger
    """
    # Create pipeline bridge
    bridge = create_alerting_pipeline(
        anomaly_detector=anomaly_detector,
        alerting_service=alerting_service,
        continuous_monitor=continuous_monitor,
        config=config,
    )

    return {
        "bridge": bridge,
        "anomaly_detector": anomaly_detector,
        "alerting_service": alerting_service,
        "continuous_monitor": continuous_monitor,
        "decision_context_manager": decision_context_manager,
        "reasoning_logger": reasoning_logger,
    }
