#!/bin/bash
#
# Auto-Resume Script for Claude Code Sessions
#
# This script automatically monitors and resumes Claude Code sessions that
# crash or disconnect during overnight autonomous development.
#
# Features:
# 1. Monitors Claude process health
# 2. Auto-restarts on crash with exponential backoff
# 3. Reads progress from claude-progress.txt to provide context
# 4. Tracks restart attempts and prevents infinite loops
# 5. Sends notifications on restart attempts
#
# Usage:
#   ./scripts/auto-resume.sh [--max-restarts N] [--backoff-base SECONDS]
#
# Configuration (via environment):
#   AUTO_RESUME_MAX_RESTARTS - Maximum restart attempts (default: 5)
#   AUTO_RESUME_BACKOFF_BASE - Base backoff seconds (default: 30)
#   AUTO_RESUME_CHECK_INTERVAL - Health check interval (default: 60)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if available
for VENV in "$PROJECT_DIR/.venv" "$PROJECT_DIR/venv"; do
    if [ -f "$VENV/bin/activate" ]; then
        source "$VENV/bin/activate"
        break
    fi
done

# Configuration
MAX_RESTARTS="${AUTO_RESUME_MAX_RESTARTS:-5}"
BACKOFF_BASE="${AUTO_RESUME_BACKOFF_BASE:-30}"
CHECK_INTERVAL="${AUTO_RESUME_CHECK_INTERVAL:-60}"
PROGRESS_FILE="$PROJECT_DIR/claude-progress.txt"
STATE_FILE="$PROJECT_DIR/logs/auto-resume-state.json"
LOG_FILE="$PROJECT_DIR/logs/auto-resume.log"

# Load progress stats via unified ProgressParser
# Part of P1-4 integration from REMEDIATION_PLAN.md
load_progress_stats() {
    if [ -f "$SCRIPT_DIR/get_progress_stats.py" ]; then
        local stats_output
        stats_output=$(python3 "$SCRIPT_DIR/get_progress_stats.py" 2>/dev/null)
        if [ -n "$stats_output" ]; then
            eval "$stats_output"
            return 0
        fi
    fi
    # Fallback to defaults if script fails
    TOTAL_TASKS=0
    COMPLETED_TASKS=0
    PENDING_TASKS=0
    COMPLETION_PCT=0
    NEXT_TASK=""
    return 1
}

# Parse arguments (called from main, not at top level)
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --max-restarts)
                MAX_RESTARTS="$2"
                shift 2
                ;;
            --backoff-base)
                BACKOFF_BASE="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [--max-restarts N] [--backoff-base SECONDS]"
                echo ""
                echo "Options:"
                echo "  --max-restarts N    Maximum restart attempts (default: 5)"
                echo "  --backoff-base SEC  Base backoff seconds (default: 30)"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date -Iseconds)
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

log_info() { log "${BLUE}INFO${NC}" "$1"; }
log_warn() { log "${YELLOW}WARN${NC}" "$1"; }
log_error() { log "${RED}ERROR${NC}" "$1"; }
log_success() { log "${GREEN}OK${NC}" "$1"; }

# Initialize state file
init_state() {
    mkdir -p "$(dirname "$STATE_FILE")"
    if [ ! -f "$STATE_FILE" ]; then
        echo '{"restart_count": 0, "last_restart": null, "session_start": null}' > "$STATE_FILE"
    fi
}

# Read state value
get_state() {
    local key="$1"
    if command -v jq &> /dev/null; then
        jq -r ".$key // \"null\"" "$STATE_FILE" 2>/dev/null || echo "null"
    else
        echo "null"
    fi
}

# Update state value
set_state() {
    local key="$1"
    local value="$2"
    if command -v jq &> /dev/null; then
        local tmp=$(mktemp)
        jq ".$key = $value" "$STATE_FILE" > "$tmp" && mv "$tmp" "$STATE_FILE"
    fi
}

# Check Claude CLI availability
# Part of P5-1 from REMEDIATION_PLAN.md
check_claude_cli() {
    if ! command -v claude &> /dev/null; then
        log_error "Claude CLI not found in PATH"
        echo "Install with: npm install -g @anthropic-ai/claude-code"
        return 1
    fi

    # Verify it responds (basic check)
    if ! claude --version &> /dev/null; then
        log_error "Claude CLI installed but not responding"
        return 1
    fi

    local version=$(claude --version 2>/dev/null || echo "unknown")
    log_info "Claude CLI: $version"
    return 0
}

# Check if Claude process is running
is_claude_running() {
    pgrep -f "claude" > /dev/null 2>&1
}

# Get Claude process PID
get_claude_pid() {
    pgrep -f "claude" 2>/dev/null | head -1
}

# Calculate backoff delay with exponential increase and jitter
# Jitter prevents "thundering herd" problem when multiple processes restart
calculate_backoff() {
    local restart_count="$1"
    local base_delay=$((BACKOFF_BASE * (2 ** restart_count)))

    # Add jitter: ±25% of base delay to prevent synchronized retries
    local jitter_range=$((base_delay / 4))
    if [ $jitter_range -gt 0 ]; then
        local jitter=$(( (RANDOM % (jitter_range * 2)) - jitter_range ))
        local delay=$((base_delay + jitter))
    else
        local delay=$base_delay
    fi

    # Cap at 10 minutes, minimum 10 seconds
    if [ $delay -gt 600 ]; then
        delay=600
    fi
    if [ $delay -lt 10 ]; then
        delay=10
    fi
    echo $delay
}

# Extract current task from progress file
# Uses unified ProgressParser via get_progress_stats.py helper
# Part of P1-4 integration from REMEDIATION_PLAN.md
get_current_task() {
    # Try to load progress stats first
    load_progress_stats

    # Return next task if available
    if [ -n "$NEXT_TASK" ]; then
        echo "$NEXT_TASK"
        return
    fi

    # Fallback: Look for Goal line
    if [ -f "$PROGRESS_FILE" ]; then
        grep "^# Goal:" "$PROGRESS_FILE" 2>/dev/null | sed 's/^# Goal: //' || echo "Continue autonomous development"
    else
        echo "Continue autonomous development"
    fi
}

# Send notification (if notify.py available)
send_notification() {
    local message="$1"
    local level="${2:-info}"

    if [ -f "$SCRIPT_DIR/notify.py" ]; then
        python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
try:
    from notify import notify_recovery_attempt, notify_recovery_result
    if '$level' == 'attempt':
        notify_recovery_attempt('$message')
    else:
        notify_recovery_result('$message', success=('$level' == 'success'))
except Exception as e:
    print(f'Notification failed: {e}', file=sys.stderr)
" 2>/dev/null || true
    fi
}

# Update progress file with restart info
update_progress() {
    local message="$1"
    if [ -f "$PROGRESS_FILE" ]; then
        echo "" >> "$PROGRESS_FILE"
        echo "## Auto-Resume Event" >> "$PROGRESS_FILE"
        echo "- Time: $(date -Iseconds)" >> "$PROGRESS_FILE"
        echo "- Event: $message" >> "$PROGRESS_FILE"
    fi
}

# Get RIC Loop state if active
get_ric_state() {
    local ric_state_file="$PROJECT_DIR/.claude/ric_state.json"
    if [ -f "$ric_state_file" ]; then
        if command -v python3 &> /dev/null; then
            python3 -c "
import json
import sys
try:
    with open('$ric_state_file') as f:
        state = json.load(f)
    if state.get('upgrade_id'):
        phase_names = {0: 'Research', 1: 'Upgrade Path', 2: 'Checklist', 3: 'Coding',
                       4: 'Double-Check', 5: 'Introspection', 6: 'Metacognition', 7: 'Integration'}
        phase = state.get('current_phase', 0)
        iteration = state.get('current_iteration', 1)
        max_iter = state.get('max_iterations', 5)
        insights = state.get('insights', [])
        p0_open = len([i for i in insights if i.get('priority') == 'P0' and i.get('status') != 'resolved'])
        p1_open = len([i for i in insights if i.get('priority') == 'P1' and i.get('status') != 'resolved'])
        p2_open = len([i for i in insights if i.get('priority') == 'P2' and i.get('status') != 'resolved'])
        print(f'''
RIC LOOP SESSION ACTIVE
=======================
Upgrade: {state.get('upgrade_id', 'Unknown')}
Iteration: {iteration}/{max_iter}
Phase: {phase} - {phase_names.get(phase, 'Unknown')}
Open Insights: P0={p0_open} P1={p1_open} P2={p2_open}

>>> RESUME AT: [ITERATION {iteration}/{max_iter}] === PHASE {phase}: {phase_names.get(phase, 'Unknown').upper()} ===

Check state: python3 .claude/hooks/ric_state_manager.py status
''')
    else:
        print('')
except Exception as e:
    print('')
"
        fi
    fi
}

# Get session state from SessionStateManager
get_session_state() {
    if [ -f "$SCRIPT_DIR/session_state_manager.py" ]; then
        python3 "$SCRIPT_DIR/session_state_manager.py" recovery 2>/dev/null || echo ""
    else
        echo ""
    fi
}

# Build resume prompt from progress file and session state
# Uses unified ProgressParser for progress stats
# Part of P1-4 integration from REMEDIATION_PLAN.md
build_resume_prompt() {
    local task=$(get_current_task)
    local restart_count=$(get_state "restart_count")
    local ric_context=$(get_ric_state)
    local session_recovery=$(get_session_state)

    # Load progress stats for detailed info
    load_progress_stats

    cat << EOF
IMPORTANT: This session has been automatically resumed after a disconnect/crash.
This is restart attempt #$restart_count of $MAX_RESTARTS.

=== PROGRESS STATUS ===
Total Tasks: ${TOTAL_TASKS:-0}
Completed: ${COMPLETED_TASKS:-0}
Pending: ${PENDING_TASKS:-0}
Completion: ${COMPLETION_PCT:-0}%
P0 Pending: ${P0_PENDING:-0} | P1 Pending: ${P1_PENDING:-0} | P2 Pending: ${P2_PENDING:-0}
=======================
$ric_context
Your NEXT task: $task

INSTRUCTIONS:
1. Read claude-progress.txt to understand what was completed
2. Read claude-recovery-context.md if it exists (session state preserved)
3. Check RIC state: python3 .claude/hooks/ric_state_manager.py status
4. Check session state: python3 scripts/session_state_manager.py status
5. Continue from where you left off
6. Do NOT repeat completed work
7. Update progress file as you work
8. Create checkpoint commits regularly

If you're unsure of the current state, review recent git commits with:
git log --oneline -10

=== SESSION STATE (if available) ===
$session_recovery
=== END SESSION STATE ===

Begin by reading the progress file and recovery context, then continue working.
EOF
}

# Start Claude session
start_claude() {
    local resume_prompt="$1"
    local session_log="$PROJECT_DIR/logs/claude-resume-$(date +%Y%m%d-%H%M%S).log"

    log_info "Starting Claude Code session..."

    cd "$PROJECT_DIR"
    claude --print "$resume_prompt" 2>&1 | tee -a "$session_log" &

    # Wait a moment for process to start
    sleep 5

    if is_claude_running; then
        log_success "Claude Code started (PID: $(get_claude_pid))"
        return 0
    else
        log_error "Claude Code failed to start"
        return 1
    fi
}

# Main monitoring loop
monitor_loop() {
    log_info "Auto-resume monitoring started"
    log_info "Max restarts: $MAX_RESTARTS"
    log_info "Backoff base: ${BACKOFF_BASE}s"
    log_info "Check interval: ${CHECK_INTERVAL}s"

    # Initialize session start time
    set_state "session_start" "\"$(date -Iseconds)\""

    while true; do
        if ! is_claude_running; then
            local restart_count=$(get_state "restart_count")
            restart_count=$((restart_count + 1))

            if [ "$restart_count" -gt "$MAX_RESTARTS" ]; then
                log_error "Maximum restart attempts ($MAX_RESTARTS) exceeded"
                update_progress "Auto-resume halted: max restarts exceeded"
                send_notification "Auto-resume halted after $MAX_RESTARTS attempts" "failure"
                exit 1
            fi

            local backoff=$(calculate_backoff $restart_count)
            log_warn "Claude process not running. Restart attempt $restart_count/$MAX_RESTARTS"
            log_info "Waiting ${backoff}s before restart (exponential backoff)"

            update_progress "Restart attempt $restart_count/$MAX_RESTARTS"
            send_notification "Restart attempt $restart_count/$MAX_RESTARTS after ${backoff}s backoff" "attempt"

            # Update state
            set_state "restart_count" "$restart_count"
            set_state "last_restart" "\"$(date -Iseconds)\""

            # Wait with backoff
            sleep "$backoff"

            # Build resume prompt and start Claude
            local prompt=$(build_resume_prompt)
            if start_claude "$prompt"; then
                log_success "Session resumed successfully"
                send_notification "Session resumed (attempt $restart_count)" "success"
            else
                log_error "Failed to restart session"
                continue
            fi
        fi

        sleep "$CHECK_INTERVAL"
    done
}

# Cleanup on exit
cleanup() {
    log_info "Auto-resume monitor shutting down"
}

trap cleanup EXIT

# Main execution
main() {
    # Parse command line arguments
    parse_args "$@"

    mkdir -p "$PROJECT_DIR/logs"

    echo ""
    echo "=============================================="
    echo "  Claude Code Auto-Resume Monitor"
    echo "=============================================="
    echo ""

    # Check Claude CLI availability before starting (P5-1)
    if ! check_claude_cli; then
        log_error "Cannot start: Claude CLI not available"
        exit 1
    fi

    init_state

    # Check if Claude is already running
    if is_claude_running; then
        log_info "Claude Code already running (PID: $(get_claude_pid))"
        log_info "Monitoring for crashes..."
    else
        log_info "Claude Code not running. Starting..."
        local task=$(get_current_task)
        local prompt="Starting autonomous session. Task: $task"

        if ! start_claude "$prompt"; then
            log_error "Could not start initial session"
            exit 1
        fi
    fi

    # Enter monitoring loop
    monitor_loop
}

# Only run main when executed directly, not when sourced
# This allows other scripts to source this file and use its functions
# (like load_progress_stats) without triggering the monitoring loop
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
