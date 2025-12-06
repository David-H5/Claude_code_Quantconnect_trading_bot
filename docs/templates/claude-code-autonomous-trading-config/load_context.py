#!/usr/bin/env python3
"""
SessionStart hook: Loads current portfolio state, market status, and recent activity.
Output is shown to Claude at session start.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_market_status():
    """Check if market is currently open."""
    now = datetime.now(timezone.utc)
    hour = now.hour
    weekday = now.weekday()

    # Simplified: NYSE hours 14:30-21:00 UTC, Mon-Fri
    if weekday >= 5:
        return "CLOSED (Weekend)"
    if 14 <= hour < 21:
        return "OPEN"
    if hour == 13 and now.minute >= 30:
        return "OPEN"
    return "CLOSED"


def load_portfolio_summary():
    """Load current portfolio state."""
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
    state_file = project_dir / "data" / "portfolio_state.json"

    if state_file.exists():
        state = json.loads(state_file.read_text())
        return {
            "cash": state.get("cash", 0),
            "positions_count": len(state.get("positions", [])),
            "total_value": state.get("total_value", state.get("cash", 0)),
            "daily_pnl": state.get("daily_pnl", 0),
        }
    return {"cash": 100000, "positions_count": 0, "total_value": 100000, "daily_pnl": 0}


def get_recent_activity():
    """Get summary of recent trading activity."""
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
    log_dir = project_dir / "logs" / "trades"

    if not log_dir.exists():
        return []

    # Get most recent log file
    log_files = sorted(log_dir.glob("trades_*.jsonl"), reverse=True)
    if not log_files:
        return []

    recent = []
    with open(log_files[0]) as f:
        for line in f:
            if line.strip():
                recent.append(json.loads(line))

    return recent[-5:]  # Last 5 entries


def main():
    trading_mode = os.environ.get("TRADING_MODE", "paper")
    market_status = get_market_status()
    portfolio = load_portfolio_summary()
    recent = get_recent_activity()

    print("=" * 60)
    print("🤖 TRADING BOT SESSION INITIALIZED")
    print("=" * 60)
    print(f"Mode: {trading_mode.upper()}")
    print(f"Market: {market_status}")
    print(f"Portfolio: ${portfolio['total_value']:,.2f} ({portfolio['positions_count']} positions)")
    print(f"Daily P&L: ${portfolio['daily_pnl']:+,.2f}")
    print("-" * 60)

    if recent:
        print("Recent activity:")
        for entry in recent[-3:]:
            symbol = entry.get("input", {}).get("symbol", "N/A")
            action = entry.get("input", {}).get("action", "N/A")
            print(f"  • {entry['timestamp'][:19]}: {action} {symbol}")

    print("-" * 60)
    print("See @docs/SAFETY.md for trading limits")
    print("Use /status for detailed portfolio view")
    print("=" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
