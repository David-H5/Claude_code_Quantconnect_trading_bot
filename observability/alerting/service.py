"""
Multi-Channel Alerting Service for Trading Bot

Provides:
- Multiple alert channels (console, email, Discord, Slack)
- Rate limiting to prevent spam
- Alert aggregation for noise reduction
- Severity-based filtering
- Integration with ErrorHandler

UPGRADE-013: Monitoring & Alerting (December 2025)
"""

from __future__ import annotations

import json
import logging
import smtplib
import threading
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


# ==============================================================================
# Alert Data Types
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


class AlertCategory(Enum):
    """Alert categories."""

    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"
    PERFORMANCE = "performance"
    ERROR = "error"
    CIRCUIT_BREAKER = "circuit_breaker"
    SERVICE_HEALTH = "service_health"
    ANOMALY = "anomaly"  # Sprint 1: Real-Time Anomaly Detection


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


# ==============================================================================
# Alert Channels
# ==============================================================================


class AlertChannel(ABC):
    """Abstract base class for alert channels."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self._send_count = 0
        self._last_send: datetime | None = None
        self._failures = 0

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send alert through this channel."""
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get channel statistics."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "send_count": self._send_count,
            "last_send": self._last_send.isoformat() if self._last_send else None,
            "failures": self._failures,
        }


class ConsoleChannel(AlertChannel):
    """Console/stdout alert channel."""

    def __init__(self, enabled: bool = True):
        super().__init__("console", enabled)

    def send(self, alert: Alert) -> bool:
        """Print alert to console."""
        if not self.enabled:
            return False

        try:
            print(alert.format_console())
            self._send_count += 1
            self._last_send = datetime.now(timezone.utc)
            return True
        except Exception as e:
            logger.error(f"Console alert failed: {e}")
            self._failures += 1
            return False


class EmailChannel(AlertChannel):
    """Email alert channel using SMTP."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        recipients: list[str] | None = None,
        from_address: str = "",
        enabled: bool = True,
    ):
        super().__init__("email", enabled)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients or []
        self.from_address = from_address or username

    def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not self.enabled or not self.recipients or not self.smtp_host:
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_address
            msg["To"] = ", ".join(self.recipients)
            msg["Subject"] = f"[{alert.severity.name}] {alert.title}"

            body = f"""
Trading Bot Alert
-----------------
Severity: {alert.severity.value.upper()}
Category: {alert.category.value}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

{alert.message}
"""
            if alert.data:
                body += "\nDetails:\n"
                for key, value in alert.data.items():
                    body += f"  {key}: {value}\n"

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)

            self._send_count += 1
            self._last_send = datetime.now(timezone.utc)
            return True

        except Exception as e:
            logger.error(f"Email alert failed: {e}")
            self._failures += 1
            return False


class DiscordChannel(AlertChannel):
    """Discord webhook alert channel."""

    def __init__(self, webhook_url: str, enabled: bool = True):
        super().__init__("discord", enabled)
        self.webhook_url = webhook_url

    def send(self, alert: Alert) -> bool:
        """Send alert to Discord webhook."""
        if not self.enabled or not self.webhook_url:
            return False

        try:
            # Discord webhook payload
            color_map = {
                AlertSeverity.DEBUG: 0x808080,
                AlertSeverity.INFO: 0x3498DB,
                AlertSeverity.WARNING: 0xF39C12,
                AlertSeverity.ERROR: 0xE74C3C,
                AlertSeverity.CRITICAL: 0x9B59B6,
            }

            embed = {
                "title": alert.title,
                "description": alert.message,
                "color": color_map.get(alert.severity, 0x3498DB),
                "fields": [
                    {"name": "Severity", "value": alert.severity.value, "inline": True},
                    {"name": "Category", "value": alert.category.value, "inline": True},
                ],
                "timestamp": alert.timestamp.isoformat(),
            }

            if alert.data:
                for key, value in list(alert.data.items())[:5]:  # Limit fields
                    embed["fields"].append(
                        {
                            "name": str(key),
                            "value": str(value)[:1024],
                            "inline": True,
                        }
                    )

            payload = {"embeds": [embed]}

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status in (200, 204):
                    self._send_count += 1
                    self._last_send = datetime.now(timezone.utc)
                    return True

            return False

        except Exception as e:
            logger.error(f"Discord alert failed: {e}")
            self._failures += 1
            return False


class SlackChannel(AlertChannel):
    """Slack webhook alert channel."""

    def __init__(self, webhook_url: str, enabled: bool = True):
        super().__init__("slack", enabled)
        self.webhook_url = webhook_url

    def send(self, alert: Alert) -> bool:
        """Send alert to Slack webhook."""
        if not self.enabled or not self.webhook_url:
            return False

        try:
            # Slack message with blocks
            emoji_map = {
                AlertSeverity.DEBUG: ":mag:",
                AlertSeverity.INFO: ":information_source:",
                AlertSeverity.WARNING: ":warning:",
                AlertSeverity.ERROR: ":x:",
                AlertSeverity.CRITICAL: ":rotating_light:",
            }

            payload = {
                "text": f"{emoji_map.get(alert.severity, ':bell:')} {alert.title}",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": alert.title},
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": alert.message},
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": (
                                    f"*Severity*: {alert.severity.value} | "
                                    f"*Category*: {alert.category.value} | "
                                    f"*Time*: {alert.timestamp.strftime('%H:%M:%S')}"
                                ),
                            }
                        ],
                    },
                ],
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    self._send_count += 1
                    self._last_send = datetime.now(timezone.utc)
                    return True

            return False

        except Exception as e:
            logger.error(f"Slack alert failed: {e}")
            self._failures += 1
            return False


# ==============================================================================
# Rate Limiting
# ==============================================================================


@dataclass
class RateLimitEntry:
    """Rate limit tracking entry."""

    key: str
    count: int = 0
    window_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RateLimiter:
    """Rate limiter for alerts."""

    def __init__(
        self,
        max_per_minute: int = 10,
        window_seconds: int = 60,
    ):
        self.max_per_minute = max_per_minute
        self.window_seconds = window_seconds
        self._entries: dict[str, RateLimitEntry] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        """Check if alert is allowed (not rate limited)."""
        with self._lock:
            now = datetime.now(timezone.utc)
            window = timedelta(seconds=self.window_seconds)

            if key not in self._entries:
                self._entries[key] = RateLimitEntry(key=key, count=1)
                return True

            entry = self._entries[key]

            # Reset window if expired
            if now - entry.window_start > window:
                entry.count = 1
                entry.window_start = now
                return True

            # Check limit
            if entry.count >= self.max_per_minute:
                return False

            entry.count += 1
            return True

    def reset(self, key: str | None = None) -> None:
        """Reset rate limit for a key or all keys."""
        with self._lock:
            if key:
                self._entries.pop(key, None)
            else:
                self._entries.clear()


# ==============================================================================
# Alert Aggregation
# ==============================================================================


@dataclass
class AggregatedAlert:
    """Aggregated alert group."""

    key: str
    first_alert: Alert
    count: int = 1
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sent: bool = False


class AlertAggregator:
    """Aggregate similar alerts to reduce noise."""

    def __init__(
        self,
        window_seconds: int = 60,
        min_count_to_aggregate: int = 3,
    ):
        self.window_seconds = window_seconds
        self.min_count = min_count_to_aggregate
        self._aggregations: dict[str, AggregatedAlert] = {}
        self._lock = threading.Lock()

    def add(self, alert: Alert) -> Alert | None:
        """
        Add alert to aggregation.

        Returns:
            Alert to send (may be aggregated summary) or None if suppressed
        """
        key = alert.aggregation_key
        now = datetime.now(timezone.utc)
        window = timedelta(seconds=self.window_seconds)

        with self._lock:
            # Clean up old aggregations
            expired = [k for k, v in self._aggregations.items() if now - v.last_seen > window]
            for k in expired:
                del self._aggregations[k]

            if key not in self._aggregations:
                # First occurrence, send immediately
                self._aggregations[key] = AggregatedAlert(
                    key=key,
                    first_alert=alert,
                    sent=True,
                )
                return alert

            agg = self._aggregations[key]
            agg.count += 1
            agg.last_seen = now

            # Only send aggregated summary at intervals
            if agg.count == self.min_count:
                # Send aggregated summary
                summary = Alert(
                    title=f"Repeated: {alert.title}",
                    message=(
                        f"This alert has occurred {agg.count} times "
                        f"in the last {self.window_seconds} seconds.\n\n"
                        f"Original: {alert.message}"
                    ),
                    severity=alert.severity,
                    category=alert.category,
                    data={"count": agg.count, **alert.data},
                    source=alert.source,
                )
                return summary

            # Suppress additional occurrences
            return None

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Get aggregation statistics."""
        with self._lock:
            return {
                key: {
                    "key": key,
                    "count": agg.count,
                    "first_seen": agg.first_seen.isoformat(),
                    "last_seen": agg.last_seen.isoformat(),
                }
                for key, agg in self._aggregations.items()
            }


# ==============================================================================
# Alerting Service
# ==============================================================================


class AlertingService:
    """
    Multi-channel alerting service with rate limiting and aggregation.

    Features:
    - Multiple alert channels (console, email, Discord, Slack)
    - Rate limiting to prevent spam
    - Alert aggregation for noise reduction
    - Severity-based filtering
    - Integration with ErrorHandler

    Example:
        >>> from config import get_config
        >>> config = get_config().get_alerting_config()
        >>> service = AlertingService(config=config)
        >>> service.send_alert(
        ...     "High Loss",
        ...     "Daily loss exceeded 2%",
        ...     severity=AlertSeverity.WARNING,
        ...     category=AlertCategory.RISK,
        ... )
    """

    def __init__(
        self,
        config: Any | None = None,
        error_handler: Any | None = None,
    ):
        """
        Initialize alerting service.

        Args:
            config: AlertingConfig from config module
            error_handler: ErrorHandler for automatic alert integration
        """
        self.config = config
        self.error_handler = error_handler

        self._channels: dict[str, AlertChannel] = {}
        self._rate_limiter: RateLimiter | None = None
        self._aggregator: AlertAggregator | None = None
        self._min_severity = AlertSeverity.WARNING
        self._alert_history: list[Alert] = []
        self._max_history = 100
        self._listeners: list[Callable[[Alert], None]] = []
        self._lock = threading.Lock()

        self._setup_from_config()
        self._connect_error_handler()

    def _setup_from_config(self) -> None:
        """Set up service from configuration."""
        if self.config is None:
            # Default: console only
            self.add_channel(ConsoleChannel(enabled=True))
            self._rate_limiter = RateLimiter(max_per_minute=10)
            self._aggregator = AlertAggregator(window_seconds=60)
            return

        # Set minimum severity
        severity_map = {
            "DEBUG": AlertSeverity.DEBUG,
            "INFO": AlertSeverity.INFO,
            "WARNING": AlertSeverity.WARNING,
            "ERROR": AlertSeverity.ERROR,
            "CRITICAL": AlertSeverity.CRITICAL,
        }
        min_sev = getattr(self.config, "min_severity", "WARNING").upper()
        self._min_severity = severity_map.get(min_sev, AlertSeverity.WARNING)

        # Console channel
        if getattr(self.config, "console_alerts", True):
            self.add_channel(ConsoleChannel(enabled=True))

        # Email channel
        if getattr(self.config, "email_enabled", False):
            self.add_channel(
                EmailChannel(
                    smtp_host=getattr(self.config, "smtp_host", ""),
                    smtp_port=getattr(self.config, "smtp_port", 587),
                    username=getattr(self.config, "smtp_username", ""),
                    password=getattr(self.config, "smtp_password", ""),
                    recipients=getattr(self.config, "email_recipients", []),
                    enabled=True,
                )
            )

        # Discord channel
        if getattr(self.config, "discord_enabled", False):
            self.add_channel(
                DiscordChannel(
                    webhook_url=getattr(self.config, "discord_webhook_url", ""),
                    enabled=True,
                )
            )

        # Slack channel
        if getattr(self.config, "slack_enabled", False):
            self.add_channel(
                SlackChannel(
                    webhook_url=getattr(self.config, "slack_webhook_url", ""),
                    enabled=True,
                )
            )

        # Rate limiter
        rate_limit = getattr(self.config, "rate_limit_per_minute", 10)
        self._rate_limiter = RateLimiter(max_per_minute=rate_limit)

        # Aggregator
        if getattr(self.config, "aggregate_similar", True):
            window = getattr(self.config, "aggregation_window_seconds", 60)
            self._aggregator = AlertAggregator(window_seconds=window)

    def _connect_error_handler(self) -> None:
        """Connect to ErrorHandler for automatic alerts."""
        if self.error_handler is None:
            return

        def on_alert(name: str, data: dict[str, Any]) -> None:
            """Handle alerts from ErrorHandler."""
            message = data.get("message", f"Alert: {name}")
            self.send_alert(
                title=name,
                message=message,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.ERROR,
                data=data,
            )

        if hasattr(self.error_handler, "add_alert_listener"):
            self.error_handler.add_alert_listener(on_alert)

    def add_channel(self, channel: AlertChannel) -> None:
        """Add an alert channel."""
        self._channels[channel.name] = channel

    def remove_channel(self, name: str) -> None:
        """Remove an alert channel."""
        self._channels.pop(name, None)

    def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        category: AlertCategory = AlertCategory.SYSTEM,
        data: dict[str, Any] | None = None,
        source: str = "",
    ) -> bool:
        """
        Send alert through all enabled channels.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            category: Alert category
            data: Additional alert data
            source: Alert source (component)

        Returns:
            True if alert was sent to at least one channel
        """
        # Check severity threshold
        if severity.level < self._min_severity.level:
            return False

        # Create alert
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            category=category,
            data=data or {},
            source=source,
        )

        # Rate limiting
        if self._rate_limiter and not self._rate_limiter.check(alert.aggregation_key):
            logger.debug(f"Alert rate limited: {title}")
            return False

        # Aggregation
        if self._aggregator:
            alert = self._aggregator.add(alert)
            if alert is None:
                return False

        # Add to history
        with self._lock:
            self._alert_history.append(alert)
            if len(self._alert_history) > self._max_history:
                self._alert_history.pop(0)

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(alert)
            except Exception:
                pass

        # Send to channels
        sent = False
        for channel in self._channels.values():
            if channel.enabled:
                try:
                    if channel.send(alert):
                        sent = True
                except Exception as e:
                    logger.error(f"Channel {channel.name} failed: {e}")

        return sent

    def send_trading_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        data: dict[str, Any] | None = None,
    ) -> bool:
        """Send a trading-related alert."""
        return self.send_alert(
            title=title,
            message=message,
            severity=severity,
            category=AlertCategory.TRADING,
            data=data,
        )

    def send_risk_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        data: dict[str, Any] | None = None,
    ) -> bool:
        """Send a risk-related alert."""
        return self.send_alert(
            title=title,
            message=message,
            severity=severity,
            category=AlertCategory.RISK,
            data=data,
        )

    def send_circuit_breaker_alert(
        self,
        reason: str,
        data: dict[str, Any] | None = None,
    ) -> bool:
        """Send circuit breaker alert."""
        return self.send_alert(
            title="Circuit Breaker Triggered",
            message=reason,
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.CIRCUIT_BREAKER,
            data=data,
        )

    def send_anomaly_alert(
        self,
        anomaly_type: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        score: float = 0.0,
        recommended_action: str = "",
        data: dict[str, Any] | None = None,
    ) -> bool:
        """
        Send anomaly detection alert.

        Sprint 1: Real-Time Anomaly Detection integration.

        Args:
            anomaly_type: Type of anomaly (flash_crash, volume_spike, etc.)
            message: Alert message
            severity: Alert severity level
            score: Anomaly score from detector
            recommended_action: Recommended action (HALT_TRADING, REDUCE_EXPOSURE, MONITOR)
            data: Additional anomaly data (feature values, thresholds, etc.)

        Returns:
            True if alert was sent to at least one channel
        """
        alert_data = data or {}
        alert_data.update(
            {
                "anomaly_type": anomaly_type,
                "anomaly_score": score,
                "recommended_action": recommended_action,
            }
        )

        return self.send_alert(
            title=f"Anomaly Detected: {anomaly_type.replace('_', ' ').title()}",
            message=message,
            severity=severity,
            category=AlertCategory.ANOMALY,
            data=alert_data,
            source="anomaly_detector",
        )

    def add_listener(self, callback: Callable[[Alert], None]) -> None:
        """Add alert listener callback."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[Alert], None]) -> None:
        """Remove alert listener callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def get_recent_alerts(
        self,
        limit: int = 10,
        severity: AlertSeverity | None = None,
        category: AlertCategory | None = None,
    ) -> list[Alert]:
        """Get recent alerts with optional filtering."""
        with self._lock:
            alerts = list(self._alert_history)

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if category:
            alerts = [a for a in alerts if a.category == category]

        return alerts[-limit:]

    def get_channel_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all channels."""
        return {name: ch.get_stats() for name, ch in self._channels.items()}

    def get_stats(self) -> dict[str, Any]:
        """Get overall alerting service stats."""
        with self._lock:
            return {
                "channels": len(self._channels),
                "enabled_channels": sum(1 for c in self._channels.values() if c.enabled),
                "total_alerts": len(self._alert_history),
                "min_severity": self._min_severity.value,
                "rate_limit_entries": (len(self._rate_limiter._entries) if self._rate_limiter else 0),
            }


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_alerting_service(
    config: Any | None = None,
    error_handler: Any | None = None,
) -> AlertingService:
    """
    Factory function to create alerting service.

    Args:
        config: AlertingConfig from config module
        error_handler: ErrorHandler for automatic integration

    Returns:
        Configured AlertingService instance
    """
    return AlertingService(
        config=config,
        error_handler=error_handler,
    )
