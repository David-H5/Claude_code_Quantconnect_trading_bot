#!/usr/bin/env python3
"""
PostToolUse hook: Logs all trading operations for audit trail.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    # Read hook input from stdin
    input_data = json.load(sys.stdin)

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})
    tool_output = input_data.get("tool_output", {})
    session_id = input_data.get("session_id", "unknown")

    # Prepare log entry
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "tool": tool_name,
        "input": tool_input,
        "output": tool_output,
        "trading_mode": os.environ.get("TRADING_MODE", "unknown"),
    }

    # Determine log file
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
    log_dir = project_dir / "logs" / "trades"
    log_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = log_dir / f"trades_{date_str}.jsonl"

    # Append to log
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Print confirmation
    symbol = tool_input.get("symbol", "N/A")
    action = tool_input.get("action", tool_input.get("side", "N/A"))
    print(f"📝 Logged: {action} {symbol} at {log_entry['timestamp']}")

    sys.exit(0)


if __name__ == "__main__":
    main()
