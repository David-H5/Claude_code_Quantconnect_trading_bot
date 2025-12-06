#!/usr/bin/env python3
"""
PreToolUse hook: Validates trading orders against risk limits.
Exit 0 = allow, Exit 2 = block with feedback to Claude.
"""

import json
import os
import sys
from pathlib import Path


# Risk limits (also defined in config/limits.yaml)
MAX_POSITION_SIZE_PCT = 0.02  # 2% of portfolio
MAX_SINGLE_ORDER_VALUE = 5000
MAX_OPEN_POSITIONS = 10
MAX_SECTOR_EXPOSURE_PCT = 0.20
BLOCKED_SYMBOLS = ["GME", "AMC"]  # High volatility exclusions


def load_portfolio_state():
    """Load current portfolio state from cache."""
    state_file = Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")) / "data" / "portfolio_state.json"
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {"cash": 100000, "positions": [], "daily_pnl": 0}


def validate_order(tool_input: dict, portfolio: dict) -> tuple[bool, str]:
    """
    Validate order against risk limits.
    Returns (is_valid, reason).
    """
    symbol = tool_input.get("symbol", "").upper()
    quantity = abs(int(tool_input.get("quantity", 0)))
    order_type = tool_input.get("order_type", "market")
    price = float(tool_input.get("price", tool_input.get("limit_price", 0)))

    # Check blocked symbols
    if symbol in BLOCKED_SYMBOLS:
        return False, f"Symbol {symbol} is blocked due to excessive volatility"

    # Check position count
    if len(portfolio.get("positions", [])) >= MAX_OPEN_POSITIONS:
        return False, f"Maximum open positions ({MAX_OPEN_POSITIONS}) reached"

    # Check order value
    order_value = quantity * price if price > 0 else quantity * 100  # Estimate
    if order_value > MAX_SINGLE_ORDER_VALUE:
        return False, f"Order value ${order_value:.2f} exceeds limit ${MAX_SINGLE_ORDER_VALUE}"

    # Check position size as percentage of portfolio
    portfolio_value = portfolio.get("cash", 100000)
    for pos in portfolio.get("positions", []):
        portfolio_value += pos.get("market_value", 0)

    position_pct = order_value / portfolio_value
    if position_pct > MAX_POSITION_SIZE_PCT:
        return False, f"Position size {position_pct:.1%} exceeds limit {MAX_POSITION_SIZE_PCT:.1%}"

    # Check for live trading mode
    if tool_input.get("live", False):
        trading_mode = os.environ.get("TRADING_MODE", "paper")
        if trading_mode != "live":
            return False, "Live trading attempted but TRADING_MODE is not 'live'"

    return True, "Order validated"


def main():
    # Read hook input from stdin
    input_data = json.load(sys.stdin)

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Load current portfolio state
    portfolio = load_portfolio_state()

    # Validate the order
    is_valid, reason = validate_order(tool_input, portfolio)

    if is_valid:
        # Allow execution - exit 0
        print(f"✅ Risk check passed: {reason}", file=sys.stdout)
        sys.exit(0)
    else:
        # Block execution - exit 2, stderr goes to Claude
        print(f"🛑 BLOCKED: {reason}", file=sys.stderr)
        print("Review risk limits in docs/SAFETY.md before retrying.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
