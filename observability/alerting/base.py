"""
Base Alerting Infrastructure

Provides abstract interfaces and common types for alert handling.

Refactored: Phase 4 - Unified Monitoring & Alerting Infrastructure
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ==============================================================================
# Enums
# ==============================================================================


class AlertSeverity(Enum):
    """Alert severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def level(self) -> int:
        """Get numeric level for comparison."""
        levels = {
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50,
        }
        return levels.get(self.value, 0)

    def __ge__(self, other: "AlertSeverity") -> bool:
        return self.level >= other.level

    def __gt__(self, other: "AlertSeverity") -> bool:
        return self.level > other.level

    def __le__(self, other: "AlertSeverity") -> bool:
        return self.level <= other.level

    def __lt__(self, other: "AlertSeverity") -> bool:
        return self.level < other.level


class AlertCategory(Enum):
    """Alert categories."""

    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"
    PERFORMANCE = "performance"
    ERROR = "error"
    CIRCUIT_BREAKER = "circuit_breaker"
    SERVICE_HEALTH = "service_health"
    ANOMALY = "anomaly"
    RESOURCE = "resource"
    STORAGE = "storage"


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class Alert:
    """Alert data structure."""

    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.INFO
    category: AlertCategory = AlertCategory.SYSTEM
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = field(default_factory=dict)
    alert_id: str = ""
    source: str = ""
    aggregation_key: str = ""

    def __post_init__(self) -> None:
        """Generate alert ID and aggregation key."""
        if not self.alert_id:
            self.alert_id = f"{self.timestamp.timestamp():.0f}-{id(self)}"
        if not self.aggregation_key:
            self.aggregation_key = f"{self.category.value}:{self.title}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
        }

    def format_console(self) -> str:
        """Format for console output."""
        timestamp = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [{self.severity.name}] {self.title}: {self.message}"

    def format_markdown(self) -> str:
        """Format as markdown for Discord/Slack."""
        emoji = {
            AlertSeverity.DEBUG: "🔍",
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨",
        }.get(self.severity, "📢")

        lines = [
            f"{emoji} **{self.title}**",
            f"*Severity*: {self.severity.value.upper()}",
            f"*Category*: {self.category.value}",
            f"*Time*: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            self.message,
        ]

        if self.data:
            lines.append("")
            lines.append("*Details*:")
            for key, value in self.data.items():
                lines.append(f"• {key}: {value}")

        return "\n".join(lines)


@dataclass
class AlertRule:
    """Rule for filtering and routing alerts."""

    name: str
    min_severity: AlertSeverity = AlertSeverity.INFO
    categories: list[AlertCategory] | None = None
    enabled: bool = True
    channels: list[str] | None = None  # Channel names to route to

    def matches(self, alert: Alert) -> bool:
        """Check if alert matches this rule."""
        if not self.enabled:
            return False

        # Check severity
        if alert.severity.level < self.min_severity.level:
            return False

        # Check category
        if self.categories and alert.category not in self.categories:
            return False

        return True


# ==============================================================================
# Abstract Base Classes
# ==============================================================================


class AbstractAlertChannel(ABC):
    """Abstract base class for alert channels."""

    def __init__(self, name: str, enabled: bool = True):
        self._name = name
        self._enabled = enabled
        self._send_count = 0
        self._last_send: datetime | None = None
        self._failures = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """
        Send alert through this channel.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get channel statistics."""
        return {
            "name": self._name,
            "enabled": self._enabled,
            "send_count": self._send_count,
            "last_send": self._last_send.isoformat() if self._last_send else None,
            "failures": self._failures,
        }

    def _record_send(self) -> None:
        """Record successful send."""
        self._send_count += 1
        self._last_send = datetime.now(timezone.utc)

    def _record_failure(self) -> None:
        """Record send failure."""
        self._failures += 1


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    category: AlertCategory = AlertCategory.SYSTEM,
    **kwargs: Any,
) -> Alert:
    """
    Create an Alert with common defaults.

    Args:
        title: Alert title
        message: Alert message
        severity: Alert severity
        category: Alert category
        **kwargs: Additional fields

    Returns:
        Alert instance
    """
    return Alert(
        title=title,
        message=message,
        severity=severity,
        category=category,
        **kwargs,
    )
