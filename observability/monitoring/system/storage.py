#!/usr/bin/env python3
"""
Storage Monitor for QuantConnect Object Store

Monitors Object Store usage, tracks quotas, and generates alerts
for the 5GB tier configuration.

Author: QuantConnect Trading Bot
Date: 2025-11-30
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class StorageAlert:
    """Alert for storage threshold breach."""

    timestamp: datetime
    severity: str  # "warning" | "critical"
    message: str
    usage_pct: float
    usage_gb: float
    limit_gb: float
    category: str | None = None


class StorageMonitor:
    """
    Monitor Object Store usage for 5GB tier.

    Tracks:
    - Total storage usage vs 5GB limit
    - Per-category quota usage
    - File count vs 50,000 limit
    - Growth trends
    - Alert generation

    Example usage:
        monitor = StorageMonitor(
            object_store_manager=store_manager,
            max_storage_gb=5,
            max_files=50000,
            alert_threshold_pct=80,
        )

        # Check usage
        stats = monitor.check_usage()

        if not monitor.is_healthy():
            alerts = monitor.get_recent_alerts()
    """

    def __init__(
        self,
        object_store_manager: object,
        max_storage_gb: float = 5,
        max_files: int = 50000,
        alert_threshold_pct: float = 80,
        critical_threshold_pct: float = 90,
        circuit_breaker: object | None = None,
        alert_callback: Callable[[StorageAlert], None] | None = None,
    ):
        """
        Initialize storage monitor.

        Args:
            object_store_manager: ObjectStoreManager instance
            max_storage_gb: Maximum storage in GB (5 for 5GB tier)
            max_files: Maximum file count (50,000 for 5GB tier)
            alert_threshold_pct: Warning threshold percentage
            critical_threshold_pct: Critical threshold percentage
            circuit_breaker: Optional circuit breaker for critical alerts
            alert_callback: Optional callback for alerts
        """
        self.store_manager = object_store_manager
        self.max_storage_gb = max_storage_gb
        self.max_files = max_files
        self.alert_threshold_pct = alert_threshold_pct
        self.critical_threshold_pct = critical_threshold_pct
        self.circuit_breaker = circuit_breaker
        self.alert_callback = alert_callback

        # Tracking
        self._alerts: list[StorageAlert] = []
        self._usage_history: list[dict] = []
        self._is_healthy = True
        self._last_cleanup = None

    def check_usage(self) -> dict:
        """
        Check current storage usage and generate alerts.

        Returns:
            Dictionary with usage statistics
        """
        stats = self.store_manager.get_storage_stats()

        # Total usage check
        total_gb = stats["total_size_mb"] / 1024
        total_pct = (total_gb / self.max_storage_gb) * 100

        # File count check
        file_count = stats["total_objects"]
        file_count_pct = (file_count / self.max_files) * 100

        # Check thresholds
        self._check_storage_thresholds(total_gb, total_pct)
        self._check_file_count_thresholds(file_count, file_count_pct)

        # Check category quotas
        for category, category_stats in stats["by_category"].items():
            self._check_category_thresholds(category, category_stats)

        # Store usage history
        usage_snapshot = {
            "timestamp": datetime.now(),
            "total_gb": total_gb,
            "total_pct": total_pct,
            "file_count": file_count,
            "by_category": stats["by_category"],
        }
        self._usage_history.append(usage_snapshot)

        # Keep last 100 snapshots
        if len(self._usage_history) > 100:
            self._usage_history = self._usage_history[-100:]

        # Return stats with additional info
        return {
            **stats,
            "total_gb": total_gb,
            "total_pct": total_pct,
            "limit_gb": self.max_storage_gb,
            "file_count": file_count,
            "file_limit": self.max_files,
            "file_count_pct": file_count_pct,
            "is_healthy": self._is_healthy,
        }

    def _check_storage_thresholds(self, usage_gb: float, usage_pct: float) -> None:
        """Check storage usage thresholds."""
        if usage_pct >= self.critical_threshold_pct:
            alert = StorageAlert(
                timestamp=datetime.now(),
                severity="critical",
                message=f"CRITICAL: Storage at {usage_pct:.1f}% ({usage_gb:.2f}/{self.max_storage_gb}GB)",
                usage_pct=usage_pct,
                usage_gb=usage_gb,
                limit_gb=self.max_storage_gb,
            )
            self._add_alert(alert)
            self._is_healthy = False

            # Trip circuit breaker
            if self.circuit_breaker:
                self.circuit_breaker.halt_all_trading(f"Object Store critical: {usage_pct:.1f}% full")

        elif usage_pct >= self.alert_threshold_pct:
            alert = StorageAlert(
                timestamp=datetime.now(),
                severity="warning",
                message=f"WARNING: Storage at {usage_pct:.1f}% ({usage_gb:.2f}/{self.max_storage_gb}GB)",
                usage_pct=usage_pct,
                usage_gb=usage_gb,
                limit_gb=self.max_storage_gb,
            )
            self._add_alert(alert)

    def _check_file_count_thresholds(self, file_count: int, file_count_pct: float) -> None:
        """Check file count thresholds."""
        if file_count_pct >= self.critical_threshold_pct:
            alert = StorageAlert(
                timestamp=datetime.now(),
                severity="critical",
                message=f"CRITICAL: File count at {file_count_pct:.1f}% ({file_count:,}/{self.max_files:,} files)",
                usage_pct=file_count_pct,
                usage_gb=file_count / self.max_files * self.max_storage_gb,
                limit_gb=self.max_storage_gb,
            )
            self._add_alert(alert)

        elif file_count_pct >= self.alert_threshold_pct:
            alert = StorageAlert(
                timestamp=datetime.now(),
                severity="warning",
                message=f"WARNING: File count at {file_count_pct:.1f}% ({file_count:,}/{self.max_files:,} files)",
                usage_pct=file_count_pct,
                usage_gb=file_count / self.max_files * self.max_storage_gb,
                limit_gb=self.max_storage_gb,
            )
            self._add_alert(alert)

    def _check_category_thresholds(self, category: str, category_stats: dict) -> None:
        """Check per-category quota thresholds."""
        size_mb = category_stats["size_mb"]
        quota_mb = category_stats["quota_mb"]

        if quota_mb > 0:
            usage_pct = (size_mb / quota_mb) * 100

            if usage_pct >= 90:
                alert = StorageAlert(
                    timestamp=datetime.now(),
                    severity="warning",
                    message=f"Category {category} at {usage_pct:.1f}% ({size_mb:.1f}/{quota_mb:.1f}MB)",
                    usage_pct=usage_pct,
                    usage_gb=size_mb / 1024,
                    limit_gb=quota_mb / 1024,
                    category=category,
                )
                self._add_alert(alert)

    def _add_alert(self, alert: StorageAlert) -> None:
        """Add alert and notify callback."""
        self._alerts.append(alert)
        logger.warning(alert.message)

        if self.alert_callback:
            self.alert_callback(alert)

        # Keep last 100 alerts
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]

    def is_healthy(self) -> bool:
        """Check if storage is healthy."""
        return self._is_healthy

    def get_recent_alerts(self, limit: int = 10) -> list[StorageAlert]:
        """Get recent storage alerts."""
        return self._alerts[-limit:]

    def get_growth_rate(self) -> float | None:
        """
        Calculate storage growth rate in GB/day.

        Returns:
            Growth rate in GB/day, or None if insufficient data
        """
        if len(self._usage_history) < 2:
            return None

        # Compare first and last snapshots
        first = self._usage_history[0]
        last = self._usage_history[-1]

        time_diff = (last["timestamp"] - first["timestamp"]).total_seconds() / 86400
        if time_diff == 0:
            return None

        size_diff = last["total_gb"] - first["total_gb"]
        growth_rate = size_diff / time_diff

        return growth_rate

    def estimate_days_until_full(self) -> int | None:
        """
        Estimate days until storage is full.

        Returns:
            Estimated days, or None if not growing
        """
        growth_rate = self.get_growth_rate()
        if not growth_rate or growth_rate <= 0:
            return None

        current_stats = self.check_usage()
        remaining_gb = self.max_storage_gb - current_stats["total_gb"]

        if remaining_gb <= 0:
            return 0

        days_until_full = remaining_gb / growth_rate
        return int(days_until_full)

    def suggest_cleanup(self) -> dict:
        """
        Suggest cleanup actions based on usage.

        Returns:
            Dictionary with cleanup suggestions
        """
        stats = self.check_usage()
        suggestions = {
            "cleanup_needed": stats["total_pct"] > self.alert_threshold_pct,
            "actions": [],
        }

        # Suggest category-specific cleanups
        for category, category_stats in stats["by_category"].items():
            quota_mb = category_stats["quota_mb"]
            if quota_mb > 0:
                usage_pct = (category_stats["size_mb"] / quota_mb) * 100

                if usage_pct > 80:
                    suggestions["actions"].append(
                        {
                            "category": category,
                            "action": "cleanup_old_files",
                            "reason": f"{usage_pct:.1f}% of quota used",
                            "expected_savings_mb": category_stats["size_mb"] * 0.2,
                        }
                    )

        # Suggest expiring old data
        if stats["total_pct"] > 70:
            suggestions["actions"].append(
                {
                    "category": "all",
                    "action": "cleanup_expired",
                    "reason": f"Total usage at {stats['total_pct']:.1f}%",
                }
            )

        return suggestions

    def get_statistics(self) -> dict:
        """
        Get detailed storage statistics.

        Returns:
            Dictionary with statistics
        """
        stats = self.check_usage()
        growth_rate = self.get_growth_rate()
        days_until_full = self.estimate_days_until_full()

        return {
            "current_usage_gb": stats["total_gb"],
            "current_usage_pct": stats["total_pct"],
            "limit_gb": self.max_storage_gb,
            "file_count": stats["file_count"],
            "file_limit": self.max_files,
            "growth_rate_gb_per_day": growth_rate,
            "days_until_full": days_until_full,
            "alerts": {
                "total": len(self._alerts),
                "critical": sum(1 for a in self._alerts if a.severity == "critical"),
                "warning": sum(1 for a in self._alerts if a.severity == "warning"),
            },
            "by_category": stats["by_category"],
        }

    def reset_health_status(self) -> None:
        """Reset health status after resolving issues."""
        self._is_healthy = True
        logger.info("Storage monitor health status reset")


def create_storage_monitor(
    object_store_manager: object,
    config: dict | None = None,
    circuit_breaker: object | None = None,
) -> StorageMonitor:
    """
    Create a configured storage monitor.

    Args:
        object_store_manager: ObjectStoreManager instance
        config: Configuration dictionary
        circuit_breaker: Optional circuit breaker instance

    Returns:
        Configured StorageMonitor
    """
    if config is None:
        config = {}

    monitoring_config = config.get("usage_monitoring", {})

    return StorageMonitor(
        object_store_manager=object_store_manager,
        max_storage_gb=config.get("max_storage_gb", 5),
        max_files=config.get("max_files", 50000),
        alert_threshold_pct=monitoring_config.get("alert_threshold_pct", 80),
        critical_threshold_pct=monitoring_config.get("critical_threshold_pct", 90),
        circuit_breaker=circuit_breaker,
    )
