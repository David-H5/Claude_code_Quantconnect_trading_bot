#!/usr/bin/env python3
"""
Notification module for watchdog.py integration.

Supports Discord and Slack webhooks for real-time alerts during
autonomous overnight Claude Code sessions.

Environment Variables:
    DISCORD_WEBHOOK_URL: Discord webhook URL for notifications
    SLACK_WEBHOOK_URL: Slack webhook URL for notifications

Usage:
    from notify import notify, notify_session_start, notify_error

    # Send generic notification
    notify("Title", "Message body", level="info")

    # Use convenience functions
    notify_session_start("Implement feature X")
    notify_error("Tests failed after 3 retries")
"""

import json
import logging
import os
from datetime import datetime


# Try to import requests, fall back to urllib
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    import urllib.error
    import urllib.request

    HAS_REQUESTS = False

# Configuration from environment
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")

# Set up logging
logger = logging.getLogger("notify")


def _post_json(url: str, data: dict, timeout: int = 10) -> bool:
    """Post JSON data to a URL, using requests or urllib."""
    try:
        json_data = json.dumps(data).encode("utf-8")

        if HAS_REQUESTS:
            response = requests.post(
                url,
                json=data,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
            return response.status_code < 400
        else:
            req = urllib.request.Request(
                url,
                data=json_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status < 400
    except Exception as e:
        logger.warning(f"Failed to post to {url}: {e}")
        return False


def send_discord(
    title: str,
    message: str,
    color: int = 0x00FF00,
    fields: list | None = None,
) -> bool:
    """
    Send notification to Discord webhook.

    Args:
        title: Embed title
        message: Embed description (max 4000 chars)
        color: Embed color as integer (0xRRGGBB)
        fields: Optional list of {"name": str, "value": str, "inline": bool}

    Returns:
        True if notification was sent successfully
    """
    if not DISCORD_WEBHOOK:
        logger.debug("Discord webhook not configured, skipping")
        return False

    embed = {
        "title": title,
        "description": message[:4000],
        "color": color,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": "Claude Code Watchdog"},
    }

    if fields:
        embed["fields"] = fields[:25]  # Discord max 25 fields

    payload = {"embeds": [embed]}

    success = _post_json(DISCORD_WEBHOOK, payload)
    if success:
        logger.debug(f"Discord notification sent: {title}")
    else:
        logger.warning(f"Discord notification failed: {title}")
    return success


def send_slack(
    title: str,
    message: str,
    color: str = "good",
    fields: list | None = None,
) -> bool:
    """
    Send notification to Slack webhook.

    Args:
        title: Attachment title
        message: Attachment text (max 3000 chars)
        color: Attachment color ("good", "warning", "danger", or hex)
        fields: Optional list of {"title": str, "value": str, "short": bool}

    Returns:
        True if notification was sent successfully
    """
    if not SLACK_WEBHOOK:
        logger.debug("Slack webhook not configured, skipping")
        return False

    attachment = {
        "color": color,
        "title": title,
        "text": message[:3000],
        "ts": int(datetime.utcnow().timestamp()),
        "footer": "Claude Code Watchdog",
    }

    if fields:
        attachment["fields"] = fields

    payload = {"attachments": [attachment]}

    success = _post_json(SLACK_WEBHOOK, payload)
    if success:
        logger.debug(f"Slack notification sent: {title}")
    else:
        logger.warning(f"Slack notification failed: {title}")
    return success


def notify(
    title: str,
    message: str,
    level: str = "info",
    fields: list | None = None,
) -> bool:
    """
    Send notification to all configured channels.

    Args:
        title: Notification title
        message: Notification body
        level: "info" (green), "warning" (yellow), "error" (red)
        fields: Optional extra fields for rich notifications

    Returns:
        True if at least one notification was sent
    """
    # Color mappings for Discord (int) and Slack (str)
    colors = {
        "info": (0x00FF00, "good"),
        "warning": (0xFFAA00, "warning"),
        "error": (0xFF0000, "danger"),
        "success": (0x00FF00, "good"),
    }
    discord_color, slack_color = colors.get(level, colors["info"])

    # Convert fields format between Discord and Slack
    discord_fields = None
    slack_fields = None

    if fields:
        discord_fields = [
            {
                "name": f.get("name", f.get("title", "")),
                "value": str(f.get("value", "")),
                "inline": f.get("inline", f.get("short", False)),
            }
            for f in fields
        ]
        slack_fields = [
            {
                "title": f.get("title", f.get("name", "")),
                "value": str(f.get("value", "")),
                "short": f.get("short", f.get("inline", False)),
            }
            for f in fields
        ]

    discord_ok = send_discord(title, message, discord_color, discord_fields)
    slack_ok = send_slack(title, message, slack_color, slack_fields)

    return discord_ok or slack_ok


# =============================================================================
# Convenience functions for watchdog.py integration
# =============================================================================


def notify_session_start(task: str, config: dict | None = None) -> None:
    """Notify that an overnight session has started."""
    fields = []
    if config:
        fields = [
            {"name": "Max Runtime", "value": f"{config.get('max_runtime_hours', 10)} hours", "inline": True},
            {"name": "Max Cost", "value": f"${config.get('max_cost_usd', 50):.2f}", "inline": True},
            {
                "name": "Checkpoint Interval",
                "value": f"{config.get('checkpoint_interval_minutes', 15)} min",
                "inline": True,
            },
        ]

    notify(
        "🚀 Overnight Session Started",
        f"**Task**: {task}\n\nWatchdog monitoring active.",
        level="info",
        fields=fields,
    )


def notify_checkpoint(phase: str, progress: str, iteration: int | None = None) -> None:
    """Notify of a checkpoint during the session."""
    title = f"📍 Checkpoint: {phase}"
    if iteration:
        title = f"📍 Checkpoint #{iteration}: {phase}"

    notify(title, progress, level="info")


def notify_context_warning(
    current_pct: float = 70,
    message: str | None = None,
    usage_pct: float | None = None,  # Deprecated alias for backwards compatibility
) -> None:
    """Notify when context window usage is high or compaction is imminent.

    Args:
        current_pct: Current context usage percentage (default: 70 for compaction)
        message: Optional custom message (for pre_compact.py hook)
        usage_pct: Deprecated, use current_pct instead
    """
    # Handle backwards compatibility
    pct = usage_pct if usage_pct is not None else current_pct

    if message:
        # Custom message (from pre_compact.py)
        notify(
            f"⚠️ Context Compaction ({pct:.0f}%)",
            message,
            level="warning",
        )
    else:
        # Standard warning
        notify(
            f"⚠️ Context at {pct:.0f}%",
            "Approaching context limit. Agent should use `/compact` or `/clear` soon.\n\n"
            "Performance degrades above 80% capacity.",
            level="warning",
        )


def notify_warning(issue: str, details: str | None = None) -> None:
    """Notify of a non-critical warning."""
    message = issue
    if details:
        message += f"\n\n```\n{details[:1000]}\n```"

    notify("⚠️ Warning", message, level="warning")


def notify_error(error: str, stack_trace: str | None = None) -> None:
    """Notify of a critical error that halted the session."""
    message = f"Session halted due to error:\n\n```\n{error[:1500]}\n```"
    if stack_trace:
        message += f"\n\n**Stack Trace**:\n```\n{stack_trace[:500]}\n```"

    notify("❌ Error - Session Halted", message, level="error")


def notify_completion(summary: str, success: bool = True, stats: dict | None = None) -> None:
    """Notify that the session has completed."""
    title = "✅ Session Complete" if success else "⚠️ Session Ended with Issues"
    level = "success" if success else "warning"

    fields = []
    if stats:
        if "runtime" in stats:
            fields.append({"name": "Runtime", "value": stats["runtime"], "inline": True})
        if "commits" in stats:
            fields.append({"name": "Commits", "value": str(stats["commits"]), "inline": True})
        if "tests_passed" in stats:
            fields.append({"name": "Tests", "value": f"{stats['tests_passed']} passed", "inline": True})

    notify(title, summary, level=level, fields=fields)


def notify_cost_alert(current: float, limit: float) -> None:
    """Notify when approaching or exceeding cost budget."""
    pct = (current / limit) * 100
    level = "warning" if pct < 90 else "error"

    notify(
        f"💰 Cost Alert: ${current:.2f} ({pct:.0f}% of limit)",
        f"Approaching ${limit:.2f} budget limit.\n\n"
        + ("**Session will halt at 100%.**" if pct >= 90 else "Monitor usage closely."),
        level=level,
        fields=[
            {"name": "Current", "value": f"${current:.2f}", "inline": True},
            {"name": "Limit", "value": f"${limit:.2f}", "inline": True},
            {"name": "Remaining", "value": f"${limit - current:.2f}", "inline": True},
        ],
    )


def notify_backtest_result(
    sharpe: float,
    drawdown: float,
    passed: bool,
    total_trades: int | None = None,
    win_rate: float | None = None,
) -> None:
    """Notify of backtest validation results."""
    status = "✅ PASSED" if passed else "❌ FAILED"
    level = "success" if passed else "warning"

    fields = [
        {"name": "Sharpe Ratio", "value": f"{sharpe:.2f}", "inline": True},
        {"name": "Max Drawdown", "value": f"{drawdown:.1%}", "inline": True},
    ]

    if total_trades is not None:
        fields.append({"name": "Trades", "value": str(total_trades), "inline": True})
    if win_rate is not None:
        fields.append({"name": "Win Rate", "value": f"{win_rate:.1%}", "inline": True})

    notify(
        f"📊 Backtest {status}",
        "Strategy validation complete.",
        level=level,
        fields=fields,
    )


def notify_rate_limit(wait_minutes: int, reason: str | None = None) -> None:
    """Notify when rate limit is hit and session is pausing."""
    message = f"Session pausing for {wait_minutes} minutes before resuming."
    if reason:
        message += f"\n\nReason: {reason}"

    notify(
        "⏳ Rate Limit Hit",
        message,
        level="warning",
        fields=[{"name": "Resume In", "value": f"{wait_minutes} minutes", "inline": True}],
    )


def notify_idle_warning(idle_minutes: float, max_idle: float) -> None:
    """Notify when session has been idle for a while."""
    notify(
        f"😴 Session Idle: {idle_minutes:.0f} min",
        f"No activity detected for {idle_minutes:.0f} minutes.\n"
        f"Session will terminate after {max_idle:.0f} minutes of inactivity.",
        level="warning",
    )


def notify_stuck_detected(messages_sample: str | None = None) -> None:
    """Notify when agent appears to be stuck in a loop."""
    message = "Agent may be stuck in a repetitive loop. Manual intervention may be required."
    if messages_sample:
        message += f"\n\n**Recent messages**:\n```\n{messages_sample[:500]}\n```"

    notify("🔄 Stuck Detection Alert", message, level="warning")


def notify_recovery_attempt(attempt: int, strategy: str) -> None:
    """Notify of automatic recovery attempt."""
    notify(
        f"🔧 Recovery Attempt #{attempt}",
        f"Attempting automatic recovery using strategy: **{strategy}**",
        level="info",
    )


def notify_recovery_result(success: bool, strategy: str, details: str | None = None) -> None:
    """Notify of recovery attempt result."""
    if success:
        notify(
            "✅ Recovery Successful",
            f"Strategy **{strategy}** succeeded. Session continuing.",
            level="success",
        )
    else:
        message = f"Strategy **{strategy}** failed."
        if details:
            message += f"\n\n{details}"
        notify("❌ Recovery Failed", message, level="error")


# =============================================================================
# Testing and CLI
# =============================================================================


def test_notifications() -> None:
    """Send test notifications to configured channels."""
    print("Testing notification channels...")
    print(f"Discord webhook: {'configured' if DISCORD_WEBHOOK else 'not set'}")
    print(f"Slack webhook: {'configured' if SLACK_WEBHOOK else 'not set'}")

    if not DISCORD_WEBHOOK and not SLACK_WEBHOOK:
        print("\nNo webhooks configured. Set environment variables:")
        print("  export DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...'")
        print("  export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...'")
        return

    print("\nSending test notification...")
    success = notify(
        "🧪 Test Notification",
        "This is a test notification from the Claude Code watchdog system.\n\n"
        "If you see this, notifications are working correctly!",
        level="info",
        fields=[
            {"name": "Test Field 1", "value": "Value 1", "inline": True},
            {"name": "Test Field 2", "value": "Value 2", "inline": True},
        ],
    )

    if success:
        print("✅ Test notification sent successfully!")
    else:
        print("❌ Test notification failed. Check webhook URLs and network.")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_notifications()
    else:
        print("Usage: python notify.py --test")
        print("\nThis module is designed to be imported by watchdog.py")
        print("Run with --test to verify webhook configuration")
