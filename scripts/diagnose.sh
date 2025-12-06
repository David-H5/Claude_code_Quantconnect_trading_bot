#!/bin/bash
#
# Pre-Overnight Diagnostic Script
#
# Performs comprehensive system and project checks before starting
# an autonomous overnight Claude Code session.
#
# Usage:
#   ./scripts/diagnose.sh [--fix]
#
# Options:
#   --fix    Attempt to fix common issues automatically
#
# Exit codes:
#   0 - All checks passed
#   1 - Some checks failed (review output)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FIX_MODE=false

if [ "$1" = "--fix" ]; then
    FIX_MODE=true
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

check_pass() { echo -e "${GREEN}✅ PASS${NC} $1"; ((PASS_COUNT++)); }
check_warn() { echo -e "${YELLOW}⚠️  WARN${NC} $1"; ((WARN_COUNT++)); }
check_fail() { echo -e "${RED}❌ FAIL${NC} $1"; ((FAIL_COUNT++)); }
check_info() { echo -e "${BLUE}ℹ️  INFO${NC} $1"; }
section() { echo -e "\n${CYAN}=== $1 ===${NC}"; }

cd "$PROJECT_DIR"

echo ""
echo "=============================================="
echo "  Claude Code Overnight Session Diagnostics"
echo "=============================================="
echo "  Project: $PROJECT_DIR"
echo "  Time: $(date)"
echo "=============================================="

# =============================================================================
section "System Requirements"
# =============================================================================

# Check Claude Code CLI
if command -v claude &> /dev/null; then
    VERSION=$(claude --version 2>/dev/null | head -1 || echo "unknown")
    check_pass "Claude Code CLI: $VERSION"
else
    check_fail "Claude Code CLI not installed"
    echo "       Install: npm install -g @anthropic-ai/claude-code"
fi

# Check Claude authentication
if claude whoami &> /dev/null 2>&1; then
    check_pass "Claude Code authenticated"
else
    check_fail "Claude Code not authenticated"
    echo "       Run: claude login"
fi

# Check Python
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    check_pass "Python: $PY_VERSION"
else
    check_fail "Python 3 not found"
fi

# Check tmux
if command -v tmux &> /dev/null; then
    TMUX_VERSION=$(tmux -V 2>&1)
    check_pass "tmux: $TMUX_VERSION"
else
    check_warn "tmux not installed (recommended for session persistence)"
    if [ "$FIX_MODE" = true ]; then
        echo "       Attempting fix: sudo apt install tmux"
        sudo apt install -y tmux 2>/dev/null && check_pass "tmux installed"
    else
        echo "       Install: sudo apt install tmux"
    fi
fi

# Check Docker (optional)
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version 2>&1 | head -1)
    check_pass "Docker: $DOCKER_VERSION"
else
    check_info "Docker not installed (optional for sandboxing)"
fi

# Check psutil
if python3 -c "import psutil" 2>/dev/null; then
    check_pass "psutil Python module available"
else
    check_warn "psutil not installed (required for watchdog)"
    if [ "$FIX_MODE" = true ]; then
        pip install psutil && check_pass "psutil installed"
    else
        echo "       Install: pip install psutil"
    fi
fi

# =============================================================================
section "API Status"
# =============================================================================

# Check Anthropic API status
if command -v curl &> /dev/null; then
    API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://api.anthropic.com/v1/messages" -H "x-api-key: test" 2>/dev/null || echo "000")
    if [ "$API_STATUS" = "401" ] || [ "$API_STATUS" = "400" ]; then
        check_pass "Anthropic API reachable (status: $API_STATUS)"
    elif [ "$API_STATUS" = "000" ]; then
        check_warn "Could not reach Anthropic API (network issue?)"
    else
        check_info "Anthropic API returned: $API_STATUS"
    fi
else
    check_info "curl not available, skipping API check"
fi

# =============================================================================
section "System Resources"
# =============================================================================

# Check disk space
DISK_USAGE=$(df -h . 2>/dev/null | tail -1 | awk '{print $5}' | tr -d '%')
DISK_AVAIL=$(df -h . 2>/dev/null | tail -1 | awk '{print $4}')
if [ -n "$DISK_USAGE" ]; then
    if [ "$DISK_USAGE" -lt 80 ]; then
        check_pass "Disk space: $DISK_AVAIL available ($DISK_USAGE% used)"
    elif [ "$DISK_USAGE" -lt 90 ]; then
        check_warn "Disk space low: $DISK_AVAIL available ($DISK_USAGE% used)"
    else
        check_fail "Disk space critical: $DISK_AVAIL available ($DISK_USAGE% used)"
    fi
fi

# Check memory
if command -v free &> /dev/null; then
    MEM_TOTAL=$(free -h | grep Mem | awk '{print $2}')
    MEM_AVAIL=$(free -h | grep Mem | awk '{print $7}')
    MEM_USED_PCT=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100}')
    if [ "$MEM_USED_PCT" -lt 70 ]; then
        check_pass "Memory: $MEM_AVAIL available of $MEM_TOTAL ($MEM_USED_PCT% used)"
    elif [ "$MEM_USED_PCT" -lt 85 ]; then
        check_warn "Memory usage elevated: $MEM_USED_PCT% used"
    else
        check_fail "Memory critical: $MEM_USED_PCT% used"
    fi
fi

# =============================================================================
section "Git Repository"
# =============================================================================

# Check git status
if git rev-parse --git-dir > /dev/null 2>&1; then
    check_pass "Git repository initialized"

    # Check for uncommitted changes
    CHANGES=$(git status --porcelain | wc -l)
    if [ "$CHANGES" -eq 0 ]; then
        check_pass "Working directory clean"
    else
        check_warn "$CHANGES uncommitted changes"
        echo "       Consider: git add -A && git commit -m 'Pre-overnight checkpoint'"
        git status --porcelain | head -5 | sed 's/^/       /'
        [ "$CHANGES" -gt 5 ] && echo "       ... and $((CHANGES - 5)) more"
    fi

    # Check current branch
    BRANCH=$(git branch --show-current 2>/dev/null || echo "detached")
    check_info "Current branch: $BRANCH"

    # Check for recent commits
    LAST_COMMIT=$(git log -1 --format="%h %s" 2>/dev/null || echo "none")
    check_info "Last commit: $LAST_COMMIT"
else
    check_fail "Not a git repository"
fi

# =============================================================================
section "Project Health"
# =============================================================================

# Check Python syntax
SYNTAX_ERRORS=0
while IFS= read -r -d '' file; do
    if ! python3 -m py_compile "$file" 2>/dev/null; then
        check_fail "Syntax error in: $file"
        ((SYNTAX_ERRORS++))
    fi
done < <(find . -name "*.py" -not -path "./.venv/*" -not -path "./.git/*" -print0 2>/dev/null)

if [ "$SYNTAX_ERRORS" -eq 0 ]; then
    check_pass "Python syntax: No errors found"
fi

# Quick test run
if [ -d "tests" ]; then
    echo -n "Running quick tests... "
    if pytest tests/ -x -q --tb=no 2>/dev/null; then
        check_pass "Tests passing"
    else
        check_warn "Some tests failing"
        echo "       Run: pytest tests/ -v for details"
    fi
else
    check_info "No tests directory found"
fi

# =============================================================================
section "Overnight Session Configuration"
# =============================================================================

# Check watchdog config
WATCHDOG_CONFIG="config/watchdog.json"
if [ -f "$WATCHDOG_CONFIG" ]; then
    check_pass "Watchdog config exists: $WATCHDOG_CONFIG"

    # Parse config values
    if command -v jq &> /dev/null; then
        MAX_RUNTIME=$(jq -r '.max_runtime_hours // 10' "$WATCHDOG_CONFIG")
        MAX_COST=$(jq -r '.max_cost_usd // 50' "$WATCHDOG_CONFIG")
        MAX_IDLE=$(jq -r '.max_idle_minutes // 30' "$WATCHDOG_CONFIG")
        check_info "  Max runtime: ${MAX_RUNTIME} hours"
        check_info "  Max cost: \$${MAX_COST}"
        check_info "  Max idle: ${MAX_IDLE} minutes"
    fi
else
    check_warn "Watchdog config not found (will use defaults)"
fi

# Check Claude settings
CLAUDE_SETTINGS=".claude/settings.json"
if [ -f "$CLAUDE_SETTINGS" ]; then
    check_pass "Claude settings exist: $CLAUDE_SETTINGS"
else
    check_warn "Claude settings not found"
fi

# Check progress file
PROGRESS_FILE="claude-progress.txt"
if [ -f "$PROGRESS_FILE" ]; then
    PROGRESS_AGE=$(( ($(date +%s) - $(stat -c %Y "$PROGRESS_FILE" 2>/dev/null || echo 0)) / 3600 ))
    if [ "$PROGRESS_AGE" -lt 24 ]; then
        check_info "Progress file exists (${PROGRESS_AGE}h old)"
    else
        check_warn "Progress file is ${PROGRESS_AGE}h old (may be stale)"
    fi
else
    check_info "No existing progress file (will be created)"
fi

# =============================================================================
section "Activity Log"
# =============================================================================

ACTIVITY_LOG="$HOME/.claude/activity-log.jsonl"
if [ -f "$ACTIVITY_LOG" ]; then
    LINES=$(wc -l < "$ACTIVITY_LOG")
    SIZE=$(du -h "$ACTIVITY_LOG" | cut -f1)
    check_info "Activity log: $LINES entries ($SIZE)"

    if [ "$LINES" -gt 10000 ]; then
        check_warn "Activity log is large, consider rotating"
        echo "       Rotate: mv $ACTIVITY_LOG ${ACTIVITY_LOG}.bak"
    fi
else
    check_info "No activity log found (normal for first run)"
fi

# =============================================================================
section "Environment Variables"
# =============================================================================

# Check for required env vars
if [ -n "$ANTHROPIC_API_KEY" ]; then
    check_pass "ANTHROPIC_API_KEY is set"
else
    check_info "ANTHROPIC_API_KEY not set (may use Claude Code auth)"
fi

if [ -n "$QC_USER_ID" ] && [ -n "$QC_API_TOKEN" ]; then
    check_pass "QuantConnect credentials set"
else
    check_warn "QuantConnect credentials not set (QC_USER_ID, QC_API_TOKEN)"
fi

if [ -n "$DISCORD_WEBHOOK_URL" ] || [ -n "$SLACK_WEBHOOK_URL" ]; then
    check_pass "Notification webhooks configured"
else
    check_info "No notification webhooks set (optional)"
fi

# =============================================================================
section "Summary"
# =============================================================================

echo ""
echo "=============================================="
echo "  Results: ${GREEN}$PASS_COUNT passed${NC}, ${YELLOW}$WARN_COUNT warnings${NC}, ${RED}$FAIL_COUNT failed${NC}"
echo "=============================================="

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Ready for overnight session!${NC}"
    echo ""
    echo "Start with:"
    echo "  ./scripts/run_overnight.sh \"Your task description\""
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}❌ Please fix failures before starting overnight session${NC}"
    echo ""
    if [ "$FIX_MODE" = false ]; then
        echo "Tip: Run with --fix to attempt automatic fixes:"
        echo "  ./scripts/diagnose.sh --fix"
    fi
    echo ""
    exit 1
fi
