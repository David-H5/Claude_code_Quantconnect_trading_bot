#!/bin/bash
#
# Strategy Validation Script for QuantConnect Algorithms
#
# Runs backtests and validates performance metrics against thresholds.
# Designed for CI/CD integration and autonomous session validation.
#
# Usage:
#   ./scripts/validate-strategy.sh [project] [min_sharpe] [max_drawdown] [start_date] [end_date]
#
# Examples:
#   ./scripts/validate-strategy.sh                          # Use defaults
#   ./scripts/validate-strategy.sh TradingBot 0.5 0.20      # Custom thresholds
#   ./scripts/validate-strategy.sh TradingBot 1.0 0.15 2023-01-01 2024-01-01
#
# Exit codes:
#   0 - Validation passed
#   1 - Validation failed
#   2 - Backtest execution failed
#

set -e

# Configuration with defaults
PROJECT="${1:-algorithms}"
MIN_SHARPE="${2:-0.5}"
MAX_DRAWDOWN="${3:-0.20}"
START_DATE="${4:-2023-01-01}"
END_DATE="${5:-2024-01-01}"
MIN_TRADES="${MIN_TRADES:-10}"
MIN_WIN_RATE="${MIN_WIN_RATE:-0.35}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; }
log_metric() { echo -e "${CYAN}[METRIC]${NC} $1"; }

echo ""
echo "=============================================="
echo "  QuantConnect Strategy Validation"
echo "=============================================="
echo ""
log_info "Project: $PROJECT"
log_info "Period: $START_DATE to $END_DATE"
log_info "Thresholds:"
echo "    Min Sharpe Ratio: $MIN_SHARPE"
echo "    Max Drawdown: $(echo "$MAX_DRAWDOWN * 100" | bc)%"
echo "    Min Trades: $MIN_TRADES"
echo "    Min Win Rate: $(echo "$MIN_WIN_RATE * 100" | bc)%"
echo ""

# Check if LEAN CLI is available
if ! command -v lean &> /dev/null; then
    log_error "LEAN CLI not found. Install with: pip install lean"
    exit 2
fi

# Run backtest
log_info "Running backtest..."
cd "$PROJECT_DIR"

BACKTEST_OUTPUT=$(mktemp)
if ! lean backtest "$PROJECT" \
    --data-provider-historical Local \
    2>&1 | tee "$BACKTEST_OUTPUT"; then
    log_error "Backtest execution failed"
    cat "$BACKTEST_OUTPUT"
    rm -f "$BACKTEST_OUTPUT"
    exit 2
fi

# Find most recent results file
RESULT_FILE=$(find "$PROJECT" -path "*backtests*" -name "*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$RESULT_FILE" ] || [ ! -f "$RESULT_FILE" ]; then
    log_error "No backtest results found"
    rm -f "$BACKTEST_OUTPUT"
    exit 2
fi

log_info "Results file: $RESULT_FILE"
echo ""

# Extract metrics using jq
extract_metric() {
    local key="$1"
    local default="$2"
    jq -r ".Statistics.\"$key\" // \"$default\"" "$RESULT_FILE" 2>/dev/null | tr -d '%$,' || echo "$default"
}

SHARPE=$(extract_metric "Sharpe Ratio" "0")
DRAWDOWN=$(extract_metric "Drawdown" "100")
TOTAL_TRADES=$(extract_metric "Total Trades" "0")
WIN_RATE=$(extract_metric "Win Rate" "0")
NET_PROFIT=$(extract_metric "Net Profit" "0")
ANNUAL_RETURN=$(extract_metric "Compounding Annual Return" "0")
SORTINO=$(extract_metric "Sortino Ratio" "0")
PROFIT_FACTOR=$(extract_metric "Profit-Loss Ratio" "0")

# Convert drawdown to decimal if it's a percentage
if [[ "$DRAWDOWN" == *"%" ]]; then
    DRAWDOWN=$(echo "$DRAWDOWN" | tr -d '%')
fi
DRAWDOWN_DECIMAL=$(echo "scale=4; $DRAWDOWN / 100" | bc)

# Convert win rate to decimal if needed
if (( $(echo "$WIN_RATE > 1" | bc -l) )); then
    WIN_RATE_DECIMAL=$(echo "scale=4; $WIN_RATE / 100" | bc)
else
    WIN_RATE_DECIMAL=$WIN_RATE
fi

# Display results
echo "=============================================="
echo "  Backtest Results"
echo "=============================================="
echo ""
log_metric "Sharpe Ratio:     $SHARPE"
log_metric "Max Drawdown:     ${DRAWDOWN}%"
log_metric "Total Trades:     $TOTAL_TRADES"
log_metric "Win Rate:         ${WIN_RATE}%"
log_metric "Net Profit:       $NET_PROFIT"
log_metric "Annual Return:    ${ANNUAL_RETURN}%"
log_metric "Sortino Ratio:    $SORTINO"
log_metric "Profit Factor:    $PROFIT_FACTOR"
echo ""

# Validation
echo "=============================================="
echo "  Validation Results"
echo "=============================================="
echo ""

PASSED=true
WARNINGS=0

# Check Sharpe Ratio
if (( $(echo "$SHARPE >= $MIN_SHARPE" | bc -l) )); then
    log_success "Sharpe Ratio: $SHARPE >= $MIN_SHARPE"
else
    log_error "Sharpe Ratio: $SHARPE < $MIN_SHARPE"
    PASSED=false
fi

# Check Drawdown
if (( $(echo "$DRAWDOWN_DECIMAL <= $MAX_DRAWDOWN" | bc -l) )); then
    log_success "Max Drawdown: ${DRAWDOWN}% <= $(echo "$MAX_DRAWDOWN * 100" | bc)%"
else
    log_error "Max Drawdown: ${DRAWDOWN}% > $(echo "$MAX_DRAWDOWN * 100" | bc)%"
    PASSED=false
fi

# Check Trade Count
if [ "$TOTAL_TRADES" -ge "$MIN_TRADES" ]; then
    log_success "Total Trades: $TOTAL_TRADES >= $MIN_TRADES"
else
    log_warn "Total Trades: $TOTAL_TRADES < $MIN_TRADES (insufficient for statistical significance)"
    ((WARNINGS++))
fi

# Check Win Rate
if (( $(echo "$WIN_RATE_DECIMAL >= $MIN_WIN_RATE" | bc -l) )); then
    log_success "Win Rate: ${WIN_RATE}% >= $(echo "$MIN_WIN_RATE * 100" | bc)%"
else
    log_warn "Win Rate: ${WIN_RATE}% < $(echo "$MIN_WIN_RATE * 100" | bc)%"
    ((WARNINGS++))
fi

# Check for positive profit
if (( $(echo "$NET_PROFIT > 0" | bc -l 2>/dev/null || echo "0") )); then
    log_success "Net Profit: $NET_PROFIT (positive)"
else
    log_warn "Net Profit: $NET_PROFIT (negative or zero)"
    ((WARNINGS++))
fi

echo ""

# Send notification if notify.py is available
if [ -f "$SCRIPT_DIR/notify.py" ]; then
    python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from notify import notify_backtest_result
notify_backtest_result(
    sharpe=$SHARPE,
    drawdown=$DRAWDOWN_DECIMAL,
    passed=$( [ "$PASSED" = true ] && echo "True" || echo "False" ),
    total_trades=$TOTAL_TRADES,
    win_rate=$WIN_RATE_DECIMAL
)
" 2>/dev/null || true
fi

# Final result
echo "=============================================="
if [ "$PASSED" = true ]; then
    if [ $WARNINGS -gt 0 ]; then
        log_success "VALIDATION PASSED (with $WARNINGS warnings)"
    else
        log_success "VALIDATION PASSED"
    fi
    rm -f "$BACKTEST_OUTPUT"
    exit 0
else
    log_error "VALIDATION FAILED"
    echo ""
    echo "To investigate:"
    echo "  1. Review backtest results: $RESULT_FILE"
    echo "  2. Check algorithm logic in $PROJECT/"
    echo "  3. Consider adjusting strategy parameters"
    rm -f "$BACKTEST_OUTPUT"
    exit 1
fi
