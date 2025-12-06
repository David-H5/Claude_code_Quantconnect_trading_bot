#!/usr/bin/env python3
"""
Resource Monitor for QuantConnect Trading Bot

Monitors system resources (RAM, CPU, latency) to ensure optimal performance
on QuantConnect compute nodes (B8-16, R8-16, L2-4).

Author: QuantConnect Trading Bot
Date: 2025-11-30
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psutil


logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Current resource usage metrics."""

    timestamp: datetime
    memory_used_mb: float
    memory_total_mb: float
    memory_pct: float
    cpu_pct: float
    active_securities: int
    active_positions: int
    broker_latency_ms: float | None = None
    order_throughput: float | None = None


@dataclass
class ResourceAlert:
    """Alert for resource threshold breach."""

    timestamp: datetime
    severity: str  # "warning" | "critical"
    metric: str
    current_value: float
    threshold: float
    message: str


class ResourceMonitor:
    """
    Monitor system resources for QuantConnect compute nodes.

    Tracks memory, CPU, and latency to ensure algorithm stays within
    node capacity. Integrates with circuit breaker for safety.

    Example usage:
        monitor = ResourceMonitor(
            memory_warning_pct=80,
            memory_critical_pct=90,
            circuit_breaker=breaker,
        )

        # In trading loop
        metrics = monitor.update(
            active_securities=len(securities),
            active_positions=len(positions),
        )

        if not monitor.is_healthy():
            # Reduce load or halt trading
            pass
    """

    def __init__(
        self,
        memory_warning_pct: float = 80,
        memory_critical_pct: float = 90,
        cpu_warning_pct: float = 75,
        cpu_critical_pct: float = 85,
        latency_warning_ms: float = 100,
        latency_critical_ms: float = 200,
        circuit_breaker: object | None = None,
        alert_callback: Callable[[ResourceAlert], None] | None = None,
        log_file: Path | None = None,
    ):
        """
        Initialize resource monitor.

        Args:
            memory_warning_pct: Warning threshold for memory usage (%)
            memory_critical_pct: Critical threshold for memory usage (%)
            cpu_warning_pct: Warning threshold for CPU usage (%)
            cpu_critical_pct: Critical threshold for CPU usage (%)
            latency_warning_ms: Warning threshold for broker latency (ms)
            latency_critical_ms: Critical threshold for broker latency (ms)
            circuit_breaker: Optional circuit breaker to trip on critical alerts
            alert_callback: Optional callback for alerts
            log_file: Path to metrics log file
        """
        self.memory_warning_pct = memory_warning_pct
        self.memory_critical_pct = memory_critical_pct
        self.cpu_warning_pct = cpu_warning_pct
        self.cpu_critical_pct = cpu_critical_pct
        self.latency_warning_ms = latency_warning_ms
        self.latency_critical_ms = latency_critical_ms
        self.circuit_breaker = circuit_breaker
        self.alert_callback = alert_callback
        self.log_file = log_file or Path("logs/resource_metrics.json")

        # Tracking
        self._metrics_history: list[ResourceMetrics] = []
        self._alerts: list[ResourceAlert] = []
        self._last_check = None
        self._check_interval = 30  # seconds
        self._is_healthy = True

        # Initialize process handle
        self._process = psutil.Process()

        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        active_securities: int = 0,
        active_positions: int = 0,
        broker_latency_ms: float | None = None,
        order_throughput: float | None = None,
    ) -> ResourceMetrics:
        """
        Update resource metrics and check thresholds.

        Args:
            active_securities: Number of active security subscriptions
            active_positions: Number of active trading positions
            broker_latency_ms: Current broker latency in milliseconds
            order_throughput: Orders per second throughput

        Returns:
            Current resource metrics
        """
        # Get memory info
        mem = psutil.virtual_memory()
        memory_used_mb = mem.used / (1024 * 1024)
        memory_total_mb = mem.total / (1024 * 1024)
        memory_pct = mem.percent

        # Get CPU usage (non-blocking)
        cpu_pct = self._process.cpu_percent(interval=0.1)

        # Create metrics snapshot
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            memory_pct=memory_pct,
            cpu_pct=cpu_pct,
            active_securities=active_securities,
            active_positions=active_positions,
            broker_latency_ms=broker_latency_ms,
            order_throughput=order_throughput,
        )

        # Store in history
        self._metrics_history.append(metrics)

        # Keep only recent history (last 1000 samples)
        if len(self._metrics_history) > 1000:
            self._metrics_history = self._metrics_history[-1000:]

        # Check thresholds
        self._check_thresholds(metrics)

        # Log metrics
        if self.log_file:
            self._log_metrics(metrics)

        self._last_check = datetime.now()

        return metrics

    def _check_thresholds(self, metrics: ResourceMetrics) -> None:
        """Check resource thresholds and generate alerts."""
        alerts = []

        # Memory checks
        if metrics.memory_pct >= self.memory_critical_pct:
            alert = ResourceAlert(
                timestamp=metrics.timestamp,
                severity="critical",
                metric="memory",
                current_value=metrics.memory_pct,
                threshold=self.memory_critical_pct,
                message=f"CRITICAL: Memory usage at {metrics.memory_pct:.1f}% " f"(limit {self.memory_critical_pct}%)",
            )
            alerts.append(alert)
            self._is_healthy = False

            # Trip circuit breaker if configured
            if self.circuit_breaker:
                self.circuit_breaker.halt_all_trading(f"Memory usage critical: {metrics.memory_pct:.1f}%")

        elif metrics.memory_pct >= self.memory_warning_pct:
            alert = ResourceAlert(
                timestamp=metrics.timestamp,
                severity="warning",
                metric="memory",
                current_value=metrics.memory_pct,
                threshold=self.memory_warning_pct,
                message=f"WARNING: Memory usage at {metrics.memory_pct:.1f}% " f"(warning {self.memory_warning_pct}%)",
            )
            alerts.append(alert)

        # CPU checks
        if metrics.cpu_pct >= self.cpu_critical_pct:
            alert = ResourceAlert(
                timestamp=metrics.timestamp,
                severity="critical",
                metric="cpu",
                current_value=metrics.cpu_pct,
                threshold=self.cpu_critical_pct,
                message=f"CRITICAL: CPU usage at {metrics.cpu_pct:.1f}% " f"(limit {self.cpu_critical_pct}%)",
            )
            alerts.append(alert)
            self._is_healthy = False

        elif metrics.cpu_pct >= self.cpu_warning_pct:
            alert = ResourceAlert(
                timestamp=metrics.timestamp,
                severity="warning",
                metric="cpu",
                current_value=metrics.cpu_pct,
                threshold=self.cpu_warning_pct,
                message=f"WARNING: CPU usage at {metrics.cpu_pct:.1f}% " f"(warning {self.cpu_warning_pct}%)",
            )
            alerts.append(alert)

        # Latency checks
        if metrics.broker_latency_ms is not None:
            if metrics.broker_latency_ms >= self.latency_critical_ms:
                alert = ResourceAlert(
                    timestamp=metrics.timestamp,
                    severity="critical",
                    metric="latency",
                    current_value=metrics.broker_latency_ms,
                    threshold=self.latency_critical_ms,
                    message=f"CRITICAL: Broker latency at {metrics.broker_latency_ms:.1f}ms "
                    f"(limit {self.latency_critical_ms}ms)",
                )
                alerts.append(alert)

            elif metrics.broker_latency_ms >= self.latency_warning_ms:
                alert = ResourceAlert(
                    timestamp=metrics.timestamp,
                    severity="warning",
                    metric="latency",
                    current_value=metrics.broker_latency_ms,
                    threshold=self.latency_warning_ms,
                    message=f"WARNING: Broker latency at {metrics.broker_latency_ms:.1f}ms "
                    f"(warning {self.latency_warning_ms}ms)",
                )
                alerts.append(alert)

        # Process alerts
        for alert in alerts:
            self._alerts.append(alert)
            logger.warning(alert.message)

            if self.alert_callback:
                self.alert_callback(alert)

        # Keep only recent alerts
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]

    def is_healthy(self) -> bool:
        """
        Check if system resources are healthy.

        Returns:
            True if all resources are within acceptable limits
        """
        return self._is_healthy

    def get_current_metrics(self) -> ResourceMetrics | None:
        """Get most recent metrics snapshot."""
        if self._metrics_history:
            return self._metrics_history[-1]
        return None

    def get_metrics_history(self, limit: int = 100) -> list[ResourceMetrics]:
        """Get recent metrics history."""
        return self._metrics_history[-limit:]

    def get_recent_alerts(self, limit: int = 10) -> list[ResourceAlert]:
        """Get recent alerts."""
        return self._alerts[-limit:]

    def estimate_memory_for_securities(self, num_securities: int) -> float:
        """
        Estimate memory usage for a given number of securities.

        Args:
            num_securities: Number of security subscriptions

        Returns:
            Estimated memory usage in MB
        """
        # Base overhead (algorithm, framework, etc.)
        base_overhead = 1100  # MB

        # Per-security overhead
        per_security = 5  # MB per security

        total_mb = base_overhead + (num_securities * per_security)
        return total_mb

    def can_add_securities(self, num_securities: int, node_ram_gb: float = 4) -> bool:
        """
        Check if we can add more securities without exceeding memory.

        Args:
            num_securities: Number of securities to add
            node_ram_gb: Node RAM capacity in GB

        Returns:
            True if adding securities won't exceed memory
        """
        current_metrics = self.get_current_metrics()
        if not current_metrics:
            return True

        estimated_additional = num_securities * 5  # MB
        estimated_total_mb = current_metrics.memory_used_mb + estimated_additional
        node_ram_mb = node_ram_gb * 1024

        # Leave 10% safety margin
        safety_margin = node_ram_mb * 0.10
        return estimated_total_mb <= (node_ram_mb - safety_margin)

    def get_statistics(self) -> dict:
        """
        Get resource usage statistics.

        Returns:
            Dictionary with min/max/avg metrics
        """
        if not self._metrics_history:
            return {}

        memory_values = [m.memory_pct for m in self._metrics_history]
        cpu_values = [m.cpu_pct for m in self._metrics_history]
        latency_values = [m.broker_latency_ms for m in self._metrics_history if m.broker_latency_ms is not None]

        stats = {
            "memory": {
                "min_pct": min(memory_values),
                "max_pct": max(memory_values),
                "avg_pct": sum(memory_values) / len(memory_values),
            },
            "cpu": {
                "min_pct": min(cpu_values),
                "max_pct": max(cpu_values),
                "avg_pct": sum(cpu_values) / len(cpu_values),
            },
            "samples": len(self._metrics_history),
            "alerts": {
                "total": len(self._alerts),
                "critical": sum(1 for a in self._alerts if a.severity == "critical"),
                "warning": sum(1 for a in self._alerts if a.severity == "warning"),
            },
        }

        if latency_values:
            stats["latency"] = {
                "min_ms": min(latency_values),
                "max_ms": max(latency_values),
                "avg_ms": sum(latency_values) / len(latency_values),
            }

        return stats

    def _log_metrics(self, metrics: ResourceMetrics) -> None:
        """Write metrics to log file."""
        try:
            log_entry = {
                "timestamp": str(metrics.timestamp),
                "memory_pct": metrics.memory_pct,
                "memory_used_mb": metrics.memory_used_mb,
                "cpu_pct": metrics.cpu_pct,
                "active_securities": metrics.active_securities,
                "active_positions": metrics.active_positions,
                "broker_latency_ms": metrics.broker_latency_ms,
                "order_throughput": metrics.order_throughput,
            }

            # Append to log file (keep last 24 hours)
            logs = []
            if self.log_file.exists():
                with open(self.log_file) as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = []

            logs.append(log_entry)

            # Keep only recent logs (e.g., last 2880 samples = 24 hours at 30s intervals)
            if len(logs) > 2880:
                logs = logs[-2880:]

            with open(self.log_file, "w") as f:
                json.dump(logs, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to write resource metrics log: {e}")

    def reset_health_status(self) -> None:
        """Reset health status (for recovery after resolving issues)."""
        self._is_healthy = True
        logger.info("Resource monitor health status reset")


def create_resource_monitor(
    config: dict | None = None,
    circuit_breaker: object | None = None,
) -> ResourceMonitor:
    """
    Create a configured resource monitor.

    Args:
        config: Configuration dictionary with resource limits
        circuit_breaker: Optional circuit breaker instance

    Returns:
        Configured ResourceMonitor instance
    """
    if config is None:
        config = {}

    return ResourceMonitor(
        memory_warning_pct=config.get("memory_warning_pct", 80),
        memory_critical_pct=config.get("memory_critical_pct", 90),
        cpu_warning_pct=config.get("cpu_warning_pct", 75),
        cpu_critical_pct=config.get("cpu_critical_pct", 85),
        latency_warning_ms=config.get("latency_warning_ms", 100),
        latency_critical_ms=config.get("latency_critical_ms", 200),
        circuit_breaker=circuit_breaker,
    )
