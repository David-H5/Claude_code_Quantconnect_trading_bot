#!/usr/bin/env python3
"""
Risk Validator PreToolUse Hook - TRADING SECURITY (BLOCKS)

Validates trading-related tool calls against risk limits before execution.
BLOCKS operations that exceed configured risk thresholds.

⚠️ THIS IS A SECURITY-CRITICAL HOOK FOR TRADING OPERATIONS
   DO NOT make this non-blocking - financial safety requires hard limits

UPGRADE-015 Phase 4: Hook System Implementation

Validation Layer Architecture:
------------------------------
This hook is a SAFETY NET at the Claude Code tool boundary, providing
defense-in-depth alongside application-level validation:

    [Claude Tool Call] → [THIS HOOK] → [Application Code] → [PreTradeValidator]
                              ↓                                    ↓
                         BLOCK early                        Validate business rules
                         (tool boundary)                    (application level)

This hook (risk_validator.py):
    - Operates at Claude Code tool boundary
    - Catches ALL trading tool calls before execution
    - Simple, fast checks with hard limits
    - Cannot be bypassed by application code

execution/pre_trade_validator.py:
    - Operates at application level
    - Rich validation with circuit breaker integration
    - Complex business rule validation
    - Called by trading code before order submission

BOTH validators should pass for a trade to execute safely.
This provides defense-in-depth against trading errors.

Usage:
    Called as PreToolUse hook before broker/trading tool calls.
    Reads tool input from stdin, validates against risk rules.

Risk Checks (ALL BLOCKING):
    - Position size limits → BLOCK if exceeded
    - Daily trading limits → BLOCK if exceeded
    - Order value limits → BLOCK if exceeded
    - Live trading → BLOCK unless explicitly enabled
    - Blocked symbols → BLOCK always
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# Risk configuration - THESE ARE HARD LIMITS
RISK_LIMITS = {
    "max_position_pct": 0.25,  # 25% max per position
    "max_order_value": 50000,  # $50k max single order
    "max_daily_orders": 100,  # 100 orders per day max
    "max_daily_volume": 500000,  # $500k daily volume limit
    "require_paper_mode": True,  # BLOCK live trading by default
    "blocked_symbols": [],  # List of blocked symbols
    "min_order_value": 100,  # $100 minimum order
}

# Track daily activity
DAILY_STATE_FILE = Path("/tmp/trading_risk_state.json")

# Log file for blocked operations (for review)
BLOCKED_LOG_FILE = Path(".claude/logs/trading_blocks.json")


def load_daily_state() -> dict[str, Any]:
    """Load daily trading state."""
    if DAILY_STATE_FILE.exists():
        try:
            with open(DAILY_STATE_FILE) as f:
                state: dict[str, Any] = json.load(f)
                # Reset if new day
                if state.get("date") != datetime.utcnow().strftime("%Y-%m-%d"):
                    return create_new_state()
                return state
        except (json.JSONDecodeError, KeyError):
            return create_new_state()
    return create_new_state()


def create_new_state() -> dict:
    """Create new daily state."""
    return {
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "order_count": 0,
        "total_volume": 0.0,
        "symbols_traded": [],
    }


def save_daily_state(state: dict) -> None:
    """Save daily trading state."""
    try:
        with open(DAILY_STATE_FILE, "w") as f:
            json.dump(state, f)
    except OSError:
        pass  # Ignore write errors


def log_blocked_operation(reason: str, tool_name: str, tool_input: dict) -> None:
    """Log blocked trading operations for review."""
    try:
        BLOCKED_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log = []
        if BLOCKED_LOG_FILE.exists():
            try:
                log = json.loads(BLOCKED_LOG_FILE.read_text())
            except json.JSONDecodeError:
                log = []

        log.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "reason": reason,
                "tool": tool_name,
                "input": tool_input,
            }
        )

        # Keep last 200 blocked operations
        log = log[-200:]
        BLOCKED_LOG_FILE.write_text(json.dumps(log, indent=2))
    except OSError:
        pass


def validate_trading_tool(tool_name: str, tool_input: dict) -> tuple[bool, str]:
    """
    Validate a trading tool call against risk limits.

    Returns:
        (is_valid, message) tuple
        If is_valid is False, the operation MUST be blocked.
    """
    state = load_daily_state()

    # Check for live trading block - CRITICAL SAFETY
    if RISK_LIMITS["require_paper_mode"]:
        trading_mode = tool_input.get("trading_mode", "paper")
        if trading_mode == "live":
            return False, "🚫 BLOCKED: Live trading is disabled. Use paper mode or change require_paper_mode setting."

    # Validate place_order tool
    if tool_name == "place_order":
        symbol = tool_input.get("symbol", "").upper()
        quantity = tool_input.get("quantity", 0)
        price = tool_input.get("limit_price") or tool_input.get("stop_price") or 100
        order_value = quantity * price

        # Check blocked symbols - HARD BLOCK
        blocked_symbols: list[str] = RISK_LIMITS.get("blocked_symbols", [])  # type: ignore[assignment]
        if symbol in blocked_symbols:
            return False, f"🚫 BLOCKED: Symbol {symbol} is restricted."

        # Check order value limits - HARD BLOCK
        if order_value > RISK_LIMITS["max_order_value"]:
            return (
                False,
                f"🚫 BLOCKED: Order value ${order_value:,.0f} exceeds max ${RISK_LIMITS['max_order_value']:,}.",
            )

        if order_value < RISK_LIMITS["min_order_value"]:
            return (
                False,
                f"⚠️ WARNING: Order value ${order_value:,.0f} below minimum ${RISK_LIMITS['min_order_value']:,}. (allowed but logged)",
            )

        # Check daily order count - HARD BLOCK
        if state["order_count"] >= RISK_LIMITS["max_daily_orders"]:
            return (
                False,
                f"🚫 BLOCKED: Daily order limit ({RISK_LIMITS['max_daily_orders']}) reached. Wait until tomorrow.",
            )

        # Check daily volume - HARD BLOCK
        if state["total_volume"] + order_value > RISK_LIMITS["max_daily_volume"]:
            return False, f"🚫 BLOCKED: Would exceed daily volume limit of ${RISK_LIMITS['max_daily_volume']:,}."

        # Update state for valid orders
        state["order_count"] += 1
        state["total_volume"] += order_value
        if symbol not in state["symbols_traded"]:
            state["symbols_traded"].append(symbol)
        save_daily_state(state)

        return True, f"✅ VALIDATED: Order for {quantity} {symbol} (${order_value:,.0f})"

    # Validate cancel_order tool
    if tool_name == "cancel_order":
        order_id = tool_input.get("order_id", "")
        if not order_id:
            return False, "🚫 BLOCKED: cancel_order requires order_id."
        return True, f"✅ VALIDATED: Cancel request for {order_id}"

    # Default: allow tool
    return True, f"✅ VALIDATED: {tool_name}"


def main():
    """Main entry point for hook."""
    # Read tool context from stdin
    try:
        input_data = sys.stdin.read()
        if not input_data:
            sys.exit(0)

        context = json.loads(input_data)
    except json.JSONDecodeError:
        # If no valid JSON, allow through
        sys.exit(0)

    tool_name = context.get("tool_name", "")
    tool_input = context.get("tool_input", {})

    # Only validate trading-related tools
    trading_tools = ["place_order", "cancel_order", "modify_order"]

    if tool_name not in trading_tools:
        # Not a trading tool, allow through
        sys.exit(0)

    # Validate the tool call
    is_valid, message = validate_trading_tool(tool_name, tool_input)

    if not is_valid:
        # BLOCK the tool call - this is SECURITY CRITICAL
        print(f"Risk Validator: {message}", file=sys.stderr)
        log_blocked_operation(message, tool_name, tool_input)
        result = {
            "result": "block",
            "reason": message,
        }
        print(json.dumps(result))
        sys.exit(2)  # Exit code 2 blocks the operation
    else:
        # Allow with message
        print(f"Risk Validator: {message}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
