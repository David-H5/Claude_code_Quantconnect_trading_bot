#!/usr/bin/env python3
"""
Trade Logger PostToolUse Hook

Logs all trading-related tool calls for audit and analysis.
Creates structured logs of orders, positions, and trading activity.

UPGRADE-015 Phase 4: Hook System Implementation

Usage:
    Called as PostToolUse hook after broker/trading tool calls.
    Appends structured log entries to trade_log.jsonl.

Logged Events:
    - Order placements
    - Order cancellations
    - Position queries
    - Fill retrievals
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


# Log file location
LOG_DIR = Path("/home/dshooter/projects/Claude_code_Quantconnect_trading_bot/logs")
TRADE_LOG_FILE = LOG_DIR / "trade_log.jsonl"


def ensure_log_dir():
    """Ensure log directory exists."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_trade_event(event: dict) -> None:
    """Append trade event to log file."""
    ensure_log_dir()

    try:
        with open(TRADE_LOG_FILE, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except OSError as e:
        print(f"Warning: Could not write trade log: {e}", file=sys.stderr)


def create_log_entry(
    tool_name: str,
    tool_input: dict,
    tool_output: dict | None,
    success: bool,
) -> dict:
    """Create structured log entry."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "tool_call",
        "tool_name": tool_name,
        "input": sanitize_input(tool_input),
        "output_summary": summarize_output(tool_output) if tool_output else None,
        "success": success,
        "session_id": os.environ.get("CLAUDE_SESSION_ID", "unknown"),
    }


def sanitize_input(tool_input: dict) -> dict:
    """Remove sensitive data from input before logging."""
    sanitized = dict(tool_input)

    # Remove any sensitive fields
    sensitive_fields = ["api_key", "token", "secret", "password", "credential"]
    for field in sensitive_fields:
        if field in sanitized:
            sanitized[field] = "[REDACTED]"

    return sanitized


def summarize_output(tool_output: dict) -> dict:
    """Create summary of tool output for logging."""
    if not tool_output:
        return {}

    summary = {}

    # Extract key fields based on tool type
    if "order_id" in tool_output:
        summary["order_id"] = tool_output["order_id"]
    if "success" in tool_output:
        summary["success"] = tool_output["success"]
    if "error" in tool_output:
        summary["error"] = tool_output["error"]
    if "status" in tool_output:
        summary["status"] = tool_output["status"]
    if "total_positions" in tool_output:
        summary["total_positions"] = tool_output["total_positions"]
    if "total_orders" in tool_output:
        summary["total_orders"] = tool_output["total_orders"]
    if "total_fills" in tool_output:
        summary["total_fills"] = tool_output["total_fills"]

    return summary


def main():
    """Main entry point for hook."""
    # Read tool context from stdin
    try:
        input_data = sys.stdin.read()
        if not input_data:
            sys.exit(0)

        context = json.loads(input_data)
    except json.JSONDecodeError:
        # If no valid JSON, exit silently
        sys.exit(0)

    tool_name = context.get("tool_name", "")
    tool_input = context.get("tool_input", {})
    tool_output = context.get("tool_output")

    # Only log trading-related tools
    trading_tools = [
        "place_order",
        "cancel_order",
        "modify_order",
        "get_positions",
        "get_orders",
        "get_fills",
        "get_account_info",
    ]

    if tool_name not in trading_tools:
        # Not a trading tool, exit
        sys.exit(0)

    # Determine success
    success = True
    if tool_output:
        if isinstance(tool_output, dict):
            success = tool_output.get("success", True)
            if "error" in tool_output:
                success = False

    # Create and save log entry
    log_entry = create_log_entry(tool_name, tool_input, tool_output, success)
    log_trade_event(log_entry)

    # Print summary to stderr for visibility
    action = "SUCCESS" if success else "FAILED"
    print(f"Trade Logger: {tool_name} {action}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
