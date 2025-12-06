"""
Alert Channels

Various channels for delivering alerts:
- console: Print to stdout
- email: SMTP email delivery
- discord: Discord webhook
- slack: Slack webhook
- webhook: Generic HTTP webhook

Usage:
    from observability.alerting.channels import (
        ConsoleChannel,
        EmailChannel,
        DiscordChannel,
        SlackChannel,
        WebhookChannel,
    )

Channels are imported from the main alerting service for backwards compatibility.
"""

# Import from service module (which has the full implementations)
from observability.alerting.service import (
    AlertChannel,
    ConsoleChannel,
    DiscordChannel,
    EmailChannel,
    SlackChannel,
)


__all__ = [
    "AlertChannel",
    "ConsoleChannel",
    "DiscordChannel",
    "EmailChannel",
    "SlackChannel",
]
