#!/bin/bash
#
# Automatic Recovery Script for Failed Autonomous Sessions
#
# Attempts to recover from failed state using multiple strategies:
# 1. Reset to last known good commit
# 2. Reset to last checkpoint tag
# 3. Clean and reinstall dependencies
#
# Usage:
#   ./scripts/auto-recover.sh [project_dir]
#
# Exit codes:
#   0 - Recovery successful
#   1 - Recovery failed, manual intervention needed
#

set -e

PROJECT_DIR="${1:-$(pwd)}"
MAX_RETRIES=3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

cd "$PROJECT_DIR"

check_health() {
    # Run quick validation to check project health
    log_info "Checking project health..."

    # Quick pytest run (fail fast)
    if pytest tests/ -x -q --tb=no 2>/dev/null; then
        return 0
    fi
    return 1
}

check_syntax() {
    # Check Python syntax errors
    log_info "Checking Python syntax..."

    local errors=0
    while IFS= read -r -d '' file; do
        if ! python3 -m py_compile "$file" 2>/dev/null; then
            log_warn "Syntax error in: $file"
            ((errors++))
        fi
    done < <(find . -name "*.py" -not -path "./.venv/*" -not -path "./.git/*" -print0)

    return $errors
}

recovery_strategy_1() {
    # Strategy 1: Reset to last good checkpoint
    log_info "Strategy 1: Reset to last known good commit"

    local last_good
    last_good=$("$SCRIPT_DIR/checkpoint.sh" last-good 2>/dev/null | head -1)

    if [ -n "$last_good" ] && [ ${#last_good} -eq 40 ]; then
        log_info "Found last good commit: ${last_good:0:8}"

        # Stash current work
        git stash push -m "Auto-recovery stash $(date +%Y%m%d_%H%M%S)" 2>/dev/null || true

        # Reset to last good
        git reset --hard "$last_good"
        return $?
    else
        log_warn "No last good commit found"
        return 1
    fi
}

recovery_strategy_2() {
    # Strategy 2: Reset to last checkpoint tag
    log_info "Strategy 2: Reset to last checkpoint tag"

    local last_checkpoint
    last_checkpoint=$(git tag -l "checkpoint-*" --sort=-creatordate | head -1)

    if [ -n "$last_checkpoint" ]; then
        log_info "Found checkpoint: $last_checkpoint"

        # Stash current work
        git stash push -m "Auto-recovery stash $(date +%Y%m%d_%H%M%S)" 2>/dev/null || true

        # Reset to checkpoint
        git reset --hard "$last_checkpoint"
        return $?
    else
        log_warn "No checkpoints found"
        return 1
    fi
}

recovery_strategy_3() {
    # Strategy 3: Clean environment and reinstall
    log_info "Strategy 3: Clean and reinstall dependencies"

    # Remove cache directories
    log_info "Removing cache directories..."
    rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache 2>/dev/null || true
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

    # Reinstall dependencies
    if [ -f "requirements.txt" ]; then
        log_info "Reinstalling dependencies..."
        pip install -r requirements.txt --quiet 2>/dev/null
    fi

    # Reinstall package in development mode
    if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
        log_info "Reinstalling package..."
        pip install -e . --quiet 2>/dev/null || true
    fi

    return 0
}

attempt_recovery() {
    local attempt=$1
    log_info "Recovery attempt $attempt of $MAX_RETRIES"

    case $attempt in
        1) recovery_strategy_1 ;;
        2) recovery_strategy_2 ;;
        3) recovery_strategy_3 ;;
        *) return 1 ;;
    esac
}

document_failure() {
    # Document the failure state for debugging
    local timestamp=$(date -Iseconds)
    local errors_file="ERRORS.md"

    {
        echo ""
        echo "## Recovery Failure - $timestamp"
        echo ""
        echo "### Git Status"
        echo '```'
        git status --short
        echo '```'
        echo ""
        echo "### Recent Commits"
        echo '```'
        git log --oneline -5
        echo '```'
        echo ""
        echo "### Test Output"
        echo '```'
        pytest tests/ -x -q --tb=short 2>&1 | tail -50
        echo '```'
    } >> "$errors_file"

    log_info "Failure documented in $errors_file"
}

notify_recovery() {
    local success=$1
    local attempt=$2

    # Try to send notification via notify.py
    if [ -f "$SCRIPT_DIR/notify.py" ]; then
        if [ "$success" = "true" ]; then
            python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from notify import notify_recovery_result
notify_recovery_result(True, 'Strategy $attempt', 'Automatic recovery successful')
" 2>/dev/null || true
        else
            python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from notify import notify_recovery_result
notify_recovery_result(False, 'All strategies', 'Manual intervention required')
" 2>/dev/null || true
        fi
    fi
}

main() {
    echo ""
    echo "=============================================="
    echo "  Automatic Recovery for Autonomous Sessions"
    echo "=============================================="
    echo ""

    # Initial health check
    if check_health; then
        log_success "Project is healthy, no recovery needed"
        exit 0
    fi

    log_warn "Project unhealthy, attempting recovery..."
    echo ""

    # Try each recovery strategy
    for i in $(seq 1 $MAX_RETRIES); do
        # Send notification about recovery attempt
        if [ -f "$SCRIPT_DIR/notify.py" ]; then
            python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from notify import notify_recovery_attempt
notify_recovery_attempt($i, 'Strategy $i')
" 2>/dev/null || true
        fi

        attempt_recovery $i

        # Check if recovery worked
        if check_health; then
            log_success "Recovery successful on attempt $i!"
            notify_recovery "true" "$i"

            # Create recovery checkpoint
            "$SCRIPT_DIR/checkpoint.sh" create "post-recovery-$(date +%Y%m%d_%H%M%S)" "Recovered via strategy $i"
            exit 0
        fi

        log_warn "Attempt $i failed, trying next strategy..."
        echo ""
    done

    # All strategies failed
    log_error "Recovery failed after $MAX_RETRIES attempts"
    log_error "Manual intervention required"
    echo ""

    document_failure
    notify_recovery "false" "0"

    echo "Suggestions:"
    echo "  1. Review ERRORS.md for failure details"
    echo "  2. Check git stash list for saved work"
    echo "  3. Run: git log --oneline -20 to see recent commits"
    echo "  4. Consider manual rollback: git reset --hard <commit>"
    echo ""

    exit 1
}

# Only run main when executed directly, not when sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi
