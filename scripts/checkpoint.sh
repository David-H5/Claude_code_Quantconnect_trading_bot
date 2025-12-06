#!/bin/bash
#
# Checkpoint Management for Autonomous Claude Code Sessions
#
# Provides git-based checkpointing with tags for easy recovery during
# overnight autonomous development sessions.
#
# Usage:
#   ./scripts/checkpoint.sh create [name] [message]  - Create named checkpoint
#   ./scripts/checkpoint.sh restore <checkpoint>     - Restore to checkpoint
#   ./scripts/checkpoint.sh list                     - List recent checkpoints
#   ./scripts/checkpoint.sh last-good                - Get last passing test commit
#   ./scripts/checkpoint.sh verify                   - Run tests before checkpoint
#   ./scripts/checkpoint.sh auto                     - Auto checkpoint with test validation
#   ./scripts/checkpoint.sh recover                  - Recover from failed session
#
# Examples:
#   ./scripts/checkpoint.sh create "pre-refactor" "Before major refactoring"
#   ./scripts/checkpoint.sh restore checkpoint-pre-refactor
#   ./scripts/checkpoint.sh auto  # Validates tests, creates checkpoint if passing
#   ./scripts/checkpoint.sh recover  # Auto-recover to last good state
#
# Environment Variables:
#   CHECKPOINT_NOTIFY=1      - Send notifications on checkpoint events
#   CHECKPOINT_SKIP_TESTS=1  - Skip test validation (use with caution)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CHECKPOINT_PREFIX="checkpoint"
PROGRESS_FILE="$PROJECT_DIR/claude-progress.txt"
TEST_LOG="$PROJECT_DIR/logs/checkpoint-tests.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

create_checkpoint() {
    local name="${1:-$(date +%Y%m%d_%H%M%S)}"
    local message="${2:-Automated checkpoint}"
    local tag_name="${CHECKPOINT_PREFIX}-${name}"

    # Stage all changes
    git add -A

    # Check if there are changes to commit
    if git diff --cached --quiet; then
        log_warn "No changes to checkpoint"
        return 0
    fi

    # Create commit with checkpoint message
    git commit -m "${CHECKPOINT_PREFIX}: ${name} - ${message}

[AI-GENERATED] Automated checkpoint for autonomous session recovery."

    # Create annotated tag
    git tag -a "${tag_name}" -m "${message}"

    log_success "Created checkpoint: ${tag_name}"
    echo "  Commit: $(git rev-parse --short HEAD)"
    echo "  Tag: ${tag_name}"
}

restore_checkpoint() {
    local checkpoint="$1"

    if [ -z "$checkpoint" ]; then
        log_error "No checkpoint specified"
        echo ""
        echo "Available checkpoints:"
        git tag -l "${CHECKPOINT_PREFIX}-*" --sort=-creatordate | head -10
        return 1
    fi

    # Add prefix if not already present
    if [[ ! "$checkpoint" == ${CHECKPOINT_PREFIX}-* ]]; then
        checkpoint="${CHECKPOINT_PREFIX}-${checkpoint}"
    fi

    # Verify checkpoint exists
    if ! git rev-parse "$checkpoint" >/dev/null 2>&1; then
        log_error "Checkpoint not found: ${checkpoint}"
        echo ""
        echo "Available checkpoints:"
        git tag -l "${CHECKPOINT_PREFIX}-*" --sort=-creatordate | head -10
        return 1
    fi

    # Stash current work
    local stash_name="Before restoring ${checkpoint} at $(date +%Y%m%d_%H%M%S)"
    if ! git diff --quiet || ! git diff --cached --quiet; then
        git stash push -m "$stash_name"
        log_warn "Current changes stashed: $stash_name"
    fi

    # Reset to checkpoint
    git reset --hard "$checkpoint"
    log_success "Restored to checkpoint: ${checkpoint}"
    echo "  Current HEAD: $(git rev-parse --short HEAD)"
    echo ""
    echo "To recover stashed changes: git stash pop"
}

list_checkpoints() {
    echo "Recent checkpoints (newest first):"
    echo "=================================="

    local count=0
    while IFS= read -r tag; do
        if [ -n "$tag" ]; then
            local commit=$(git rev-list -n 1 "$tag" 2>/dev/null)
            local date=$(git log -1 --format="%ci" "$commit" 2>/dev/null | cut -d' ' -f1,2)
            local message=$(git tag -l -n1 "$tag" | sed "s/^${tag}[[:space:]]*//" )
            printf "  %-30s  %s  %s\n" "$tag" "$date" "$message"
            ((count++))
        fi
    done < <(git tag -l "${CHECKPOINT_PREFIX}-*" --sort=-creatordate | head -20)

    if [ $count -eq 0 ]; then
        echo "  (no checkpoints found)"
    fi

    echo ""
    echo "Total: $count checkpoints"
}

get_last_good() {
    # Find last commit where tests passed
    # Look for commits with "test: passing" or successful test indicators
    local last_good

    # Strategy 1: Look for explicit "test: passing" commits
    last_good=$(git log --oneline --all --grep="test: passing" -1 --format="%H" 2>/dev/null)

    if [ -z "$last_good" ]; then
        # Strategy 2: Look for any checkpoint commit
        last_good=$(git log --oneline --all --grep="${CHECKPOINT_PREFIX}:" -1 --format="%H" 2>/dev/null)
    fi

    if [ -z "$last_good" ]; then
        # Strategy 3: Look for most recent passing CI
        last_good=$(git log --oneline --all --grep="\[CI PASS\]" -1 --format="%H" 2>/dev/null)
    fi

    if [ -n "$last_good" ]; then
        echo "$last_good"
        log_success "Last known good commit: $(git log -1 --format='%h - %s' "$last_good")"
    else
        log_warn "No known good commit found"
        echo "Tip: Run tests and commit with 'test: passing' in message to track good states"
        return 1
    fi
}

quick_checkpoint() {
    # Create a quick checkpoint with auto-generated name based on current activity
    local name="quick-$(date +%H%M%S)"
    local message="Quick checkpoint during autonomous session"

    # Try to get context from progress file
    if [ -f "claude-progress.txt" ]; then
        local current_task=$(grep -m1 "^\- \[ \]" claude-progress.txt 2>/dev/null | sed 's/^- \[ \] //')
        if [ -n "$current_task" ]; then
            message="Checkpoint during: $current_task"
        fi
    fi

    create_checkpoint "$name" "$message"
}

show_diff_since_checkpoint() {
    local checkpoint="$1"

    if [ -z "$checkpoint" ]; then
        # Use most recent checkpoint
        checkpoint=$(git tag -l "${CHECKPOINT_PREFIX}-*" --sort=-creatordate | head -1)
    fi

    if [ -z "$checkpoint" ]; then
        log_error "No checkpoints found"
        return 1
    fi

    echo "Changes since ${checkpoint}:"
    echo "=============================="
    git diff --stat "$checkpoint" HEAD
    echo ""
    git diff "$checkpoint" HEAD --name-only | head -20
}

# Send notification if configured
send_notification() {
    local message="$1"
    local level="${2:-info}"

    if [ "${CHECKPOINT_NOTIFY:-0}" = "1" ] && [ -f "$SCRIPT_DIR/notify.py" ]; then
        python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
try:
    from notify import notify_checkpoint
    notify_checkpoint('$message', 0)
except Exception as e:
    print(f'Notification failed: {e}', file=sys.stderr)
" 2>/dev/null || true
    fi
}

# Verify tests before checkpoint
verify_tests() {
    local quick="${1:-false}"

    mkdir -p "$(dirname "$TEST_LOG")"
    echo "Running test verification..."

    local test_cmd="pytest tests/ -x -q --tb=short"
    if [ "$quick" = "true" ]; then
        test_cmd="pytest tests/ -x -q --tb=line -m 'not slow'"
    fi

    if $test_cmd > "$TEST_LOG" 2>&1; then
        log_success "Tests passed"
        echo "PASS" >> "$TEST_LOG"
        return 0
    else
        log_error "Tests failed - see $TEST_LOG"
        tail -20 "$TEST_LOG"
        return 1
    fi
}

# Auto checkpoint with test validation
auto_checkpoint() {
    local name="${1:-auto-$(date +%Y%m%d_%H%M%S)}"

    # Check for changes
    if git diff --quiet && git diff --cached --quiet; then
        log_warn "No changes to checkpoint"
        return 0
    fi

    # Skip tests if requested
    if [ "${CHECKPOINT_SKIP_TESTS:-0}" = "1" ]; then
        log_warn "Skipping test validation (CHECKPOINT_SKIP_TESTS=1)"
        create_checkpoint "$name" "Auto checkpoint (tests skipped)"
        return $?
    fi

    # Run quick tests
    echo "Validating tests before checkpoint..."
    if verify_tests "true"; then
        create_checkpoint "$name" "Auto checkpoint (tests passing)"
        send_notification "Checkpoint created: $name (tests passing)" "success"
        return 0
    else
        log_error "Tests failing - checkpoint not created"
        log_warn "Use CHECKPOINT_SKIP_TESTS=1 to force checkpoint"
        send_notification "Checkpoint blocked: tests failing" "warning"
        return 1
    fi
}

# Recover from failed session
recover_session() {
    echo "Starting session recovery..."
    echo "============================"

    # 1. Find last good state
    local last_good=$(get_last_good 2>/dev/null)

    if [ -z "$last_good" ]; then
        log_warn "No known good commit found"
        echo ""
        echo "Available checkpoints:"
        git tag -l "${CHECKPOINT_PREFIX}-*" --sort=-creatordate | head -5
        echo ""
        echo "Options:"
        echo "  1. Restore to a checkpoint: $0 restore <checkpoint>"
        echo "  2. Check git stash: git stash list"
        echo "  3. View recent commits: git log --oneline -10"
        return 1
    fi

    echo ""
    echo "Last known good state: $(git log -1 --format='%h - %s' "$last_good")"
    echo ""

    # 2. Check current test status
    echo "Checking current test status..."
    if pytest tests/ -x -q --tb=line > /dev/null 2>&1; then
        log_success "Current state is healthy - no recovery needed"
        return 0
    fi

    log_warn "Current tests failing - recommending recovery"
    echo ""

    # 3. Show changes since last good
    local changes=$(git rev-list --count "$last_good"..HEAD 2>/dev/null || echo "0")
    echo "Commits since last good: $changes"

    if [ "$changes" -gt 0 ]; then
        echo ""
        echo "Recent commits:"
        git log --oneline "$last_good"..HEAD | head -5
    fi

    echo ""
    echo "Recovery options:"
    echo "  1. Soft reset (keep changes): git reset --soft $last_good"
    echo "  2. Hard reset (discard changes): git reset --hard $last_good"
    echo "  3. Create branch from current: git checkout -b broken-$(date +%Y%m%d)"
    echo ""
    echo "Recommended: Create branch first, then hard reset to last good"

    # Update progress file
    if [ -f "$PROGRESS_FILE" ]; then
        echo "" >> "$PROGRESS_FILE"
        echo "## Recovery Check - $(date -Iseconds)" >> "$PROGRESS_FILE"
        echo "- Last good: $last_good" >> "$PROGRESS_FILE"
        echo "- Commits since: $changes" >> "$PROGRESS_FILE"
    fi

    send_notification "Recovery check: $changes commits since last good state" "warning"
}

# Periodic auto-checkpoint (called by cron/watchdog)
periodic_checkpoint() {
    local interval_minutes="${1:-30}"
    local state_file="$PROJECT_DIR/logs/last-checkpoint.txt"

    mkdir -p "$(dirname "$state_file")"

    # Check if enough time has passed
    if [ -f "$state_file" ]; then
        local last_checkpoint=$(cat "$state_file")
        local now=$(date +%s)
        local elapsed=$(( (now - last_checkpoint) / 60 ))

        if [ "$elapsed" -lt "$interval_minutes" ]; then
            echo "Last checkpoint ${elapsed}m ago (interval: ${interval_minutes}m)"
            return 0
        fi
    fi

    # Check for changes
    if git diff --quiet && git diff --cached --quiet; then
        echo "No changes since last checkpoint"
        return 0
    fi

    # Create checkpoint
    auto_checkpoint "periodic-$(date +%H%M)"

    # Update state file
    date +%s > "$state_file"
}

# Main command router
case "${1:-}" in
    create)
        create_checkpoint "$2" "$3"
        ;;
    restore)
        restore_checkpoint "$2"
        ;;
    list)
        list_checkpoints
        ;;
    last-good)
        get_last_good
        ;;
    quick)
        quick_checkpoint
        ;;
    diff)
        show_diff_since_checkpoint "$2"
        ;;
    verify)
        verify_tests "${2:-false}"
        ;;
    auto)
        auto_checkpoint "$2"
        ;;
    recover)
        recover_session
        ;;
    periodic)
        periodic_checkpoint "${2:-30}"
        ;;
    *)
        echo "Checkpoint Management for Autonomous Sessions"
        echo ""
        echo "Usage: $0 <command> [args]"
        echo ""
        echo "Commands:"
        echo "  create [name] [message]   Create named checkpoint"
        echo "  restore <checkpoint>      Restore to checkpoint (stashes current work)"
        echo "  list                      List recent checkpoints"
        echo "  last-good                 Get last commit with passing tests"
        echo "  quick                     Create quick checkpoint with auto-name"
        echo "  diff [checkpoint]         Show changes since checkpoint"
        echo "  verify [quick]            Run test verification"
        echo "  auto [name]               Auto checkpoint with test validation"
        echo "  recover                   Start recovery from failed session"
        echo "  periodic [minutes]        Periodic checkpoint (for cron)"
        echo ""
        echo "Environment Variables:"
        echo "  CHECKPOINT_NOTIFY=1       Send notifications on events"
        echo "  CHECKPOINT_SKIP_TESTS=1   Skip test validation"
        echo ""
        echo "Examples:"
        echo "  $0 create pre-refactor 'Before major changes'"
        echo "  $0 restore checkpoint-pre-refactor"
        echo "  $0 auto                    # Test-validated checkpoint"
        echo "  $0 recover                 # Find/restore last good state"
        echo "  $0 periodic 30             # Checkpoint if 30+ min passed"
        ;;
esac
