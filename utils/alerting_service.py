"""
Multi-Channel Alerting Service for Trading Bot

DEPRECATED: This module has been moved to observability.alerting.service.
This file provides backwards-compatible re-exports.

Use the new import path:
    from observability.alerting import AlertingService, Alert, AlertSeverity

Original: Multi-channel alert delivery
Refactored: Phase 4 - Unified Monitoring & Alerting Infrastructure
"""

# Re-export everything from new location for backwards compatibility
from observability.alerting.service import (
    # Data classes
    Alert,
    AlertAggregator,
    AlertCategory,
    # Channels
    AlertChannel,
    # Service
    AlertingService,
    # Enums
    AlertSeverity,
    ConsoleChannel,
    DiscordChannel,
    EmailChannel,
    # Rate limiting
    RateLimiter,
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
    # Channels
    "AlertChannel",
    "ConsoleChannel",
    "EmailChannel",
    "DiscordChannel",
    "SlackChannel",
    # Service
    "AlertingService",
    # Rate limiting
    "RateLimiter",
    "AlertAggregator",
    # Factory
    "create_alerting_service",
]
