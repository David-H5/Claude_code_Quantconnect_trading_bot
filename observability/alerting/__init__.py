"""
Unified Alerting Infrastructure

Provides consolidated alerting for the trading bot:
- Base alerting interfaces (AbstractAlertChannel)
- Alert data types (Alert, AlertSeverity, AlertCategory)
- Alert channels (console, email, Discord, Slack, webhook)
- Alert service with rate limiting and aggregation

Usage:
    # Create alerting service
    from observability.alerting import (
        AlertingService,
        Alert,
        AlertSeverity,
        AlertCategory,
        create_alerting_service,
    )

    # Create service with channels
    service = create_alerting_service(
        console_enabled=True,
        email_enabled=False,
    )

    # Send alert
    service.send_alert(
        title="High Memory Usage",
        message="Memory usage exceeded 80%",
        severity=AlertSeverity.WARNING,
        category=AlertCategory.RESOURCE,
    )

    # Or with Alert object
    from observability.alerting import create_alert
    alert = create_alert(
        title="Circuit Breaker Triggered",
        message="Trading halted due to daily loss limit",
        severity=AlertSeverity.CRITICAL,
        category=AlertCategory.CIRCUIT_BREAKER,
    )
    service.send(alert)

Refactored: Phase 4 - Unified Monitoring & Alerting Infrastructure
"""

# Base types
from observability.alerting.base import (
    # Abstract base
    AbstractAlertChannel,
    # Data classes
    Alert,
    AlertCategory,
    AlertRule,
    # Enums
    AlertSeverity,
    # Factory functions
    create_alert,
)

# Full service
from observability.alerting.service import (
    # Channels
    AlertChannel,
    # Service
    AlertingService,
    ConsoleChannel,
    DiscordChannel,
    EmailChannel,
    SlackChannel,
    # Factory
    create_alerting_service,
)


__all__ = [
    # Enums
    "AlertSeverity",
    "AlertCategory",
    # Data classes
    "Alert",
    "AlertRule",
    # Abstract base
    "AbstractAlertChannel",
    # Channels
    "AlertChannel",
    "ConsoleChannel",
    "EmailChannel",
    "DiscordChannel",
    "SlackChannel",
    # Service
    "AlertingService",
    # Factory functions
    "create_alert",
    "create_alerting_service",
]
