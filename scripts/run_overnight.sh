#!/bin/bash
#
# Overnight Autonomous Claude Code Session Startup Script
#
# This script:
# 1. Ensures session runs in tmux for persistence
# 2. Runs pre-flight checks
# 3. Initializes progress file
# 4. Starts watchdog process
# 5. Starts Claude Code session
# 6. Monitors and logs everything
#
# Usage:
#   ./scripts/run_overnight.sh [task_description]
#   ./scripts/run_overnight.sh --model opus "Complex refactoring task"
#   ./scripts/run_overnight.sh --with-recovery "Feature implementation"
#
# Options:
#   --model [sonnet|opus]  - Select model (default: sonnet)
#   --with-recovery        - Enable auto-resume on crash
#   --continuous           - Enable continuous mode (blocks stop until tasks done)
#   --no-watchdog          - Disable watchdog (for debugging)
#   --duration HOURS       - Max session duration (default: 10)
#
# Example:
#   ./scripts/run_overnight.sh "Implement two-part spread execution improvements"
#   ./scripts/run_overnight.sh --model opus --with-recovery "Major refactoring"
#
# Attach to session:
#   tmux attach -t overnight-dev
#
# Schedule overnight sessions:
#   crontab -e
#   # Run at 11 PM on weekdays
#   0 23 * * 1-5 cd /path/to/project && ./scripts/run_overnight.sh "Continue feature work"
#

set -e

# =============================================================================
# TMUX SESSION PERSISTENCE WRAPPER
# =============================================================================
# Ensures overnight session survives terminal disconnections and VS Code restarts
SESSION_NAME="overnight-dev"

if [ -z "$TMUX" ]; then
    # Not inside tmux - create or attach to session
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        # Create new session and run this script inside it
        tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)"
        tmux send-keys -t "$SESSION_NAME" "$0 $*" C-m
        echo ""
        echo "=============================================="
        echo "  Started in tmux session: $SESSION_NAME"
        echo "=============================================="
        echo ""
        echo "Commands:"
        echo "  Attach:  tmux attach -t $SESSION_NAME"
        echo "  Detach:  Ctrl+b, then d"
        echo "  List:    tmux ls"
        echo "  Kill:    tmux kill-session -t $SESSION_NAME"
        echo ""
        echo "Session will persist even if terminal closes."
        echo ""
        exit 0
    else
        # Session exists, attach to it
        echo "Session '$SESSION_NAME' already running."
        echo "Attaching... (Ctrl+b, d to detach)"
        tmux attach -t "$SESSION_NAME"
        exit 0
    fi
fi

# If we reach here, we're inside tmux - continue with normal execution
# =============================================================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if available
VENV_PATHS=("$PROJECT_DIR/.venv" "$PROJECT_DIR/venv")
for VENV in "${VENV_PATHS[@]}"; do
    if [ -f "$VENV/bin/activate" ]; then
        source "$VENV/bin/activate"
        echo "Activated virtual environment: $VENV"
        break
    fi
done
LOGS_DIR="$PROJECT_DIR/logs"
PROGRESS_FILE="$PROJECT_DIR/claude-progress.txt"
WATCHDOG_CONFIG="$PROJECT_DIR/config/watchdog.json"
SESSION_ID="$(date +%Y%m%d-%H%M%S)"

# Load configuration from overnight.yaml via Python wrapper
# Part of P1-2 integration from REMEDIATION_PLAN.md
if [ -f "$SCRIPT_DIR/load_overnight_config.py" ]; then
    CONFIG_OUTPUT=$(python3 "$SCRIPT_DIR/load_overnight_config.py" 2>/dev/null)
    if [ -n "$CONFIG_OUTPUT" ]; then
        eval "$CONFIG_OUTPUT"
        echo "[CONFIG] Loaded overnight.yaml configuration"
    fi
fi

# Default options (fallbacks if config not loaded, or overridden by CLI args)
MODEL="sonnet"
WITH_RECOVERY=false
ENABLE_WATCHDOG=true
# Use CONTINUOUS_MODE from config if set, otherwise default to false
CONTINUOUS_MODE="${CONTINUOUS_MODE:-0}"
[ "$CONTINUOUS_MODE" = "1" ] && CONTINUOUS_MODE=true || CONTINUOUS_MODE=false
# Use MAX_RUNTIME_HOURS from config if set, otherwise default to 10
MAX_DURATION_HOURS="${MAX_RUNTIME_HOURS:-10}"
USE_DYNAMIC_ROUTING=true  # UPGRADE-012: Use hierarchical prompt system
# Use RIC_MODE from config if set, otherwise default to SUGGESTED
RIC_MODE="${RIC_MODE:-SUGGESTED}"  # UPGRADE-016: RIC Loop enforcement (ENFORCED | SUGGESTED | DISABLED)
AUTO_RIC_INIT=""  # Auto-initialize RIC session with this upgrade ID

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --with-recovery)
            WITH_RECOVERY=true
            shift
            ;;
        --no-watchdog)
            ENABLE_WATCHDOG=false
            shift
            ;;
        --continuous)
            CONTINUOUS_MODE=true
            shift
            ;;
        --duration)
            MAX_DURATION_HOURS="$2"
            shift 2
            ;;
        --no-routing)
            USE_DYNAMIC_ROUTING=false
            shift
            ;;
        --ric-mode)
            RIC_MODE="$2"
            shift 2
            ;;
        --ric-init)
            AUTO_RIC_INIT="$2"
            shift 2
            ;;
        --help)
            echo "Overnight Autonomous Session Launcher"
            echo ""
            echo "Usage: $0 [options] [task_description]"
            echo ""
            echo "Options:"
            echo "  --model [sonnet|opus]  Model to use (default: sonnet)"
            echo "  --with-recovery        Enable auto-resume on crash"
            echo "  --continuous           Enable continuous mode (blocks stop until tasks done)"
            echo "  --no-watchdog          Disable watchdog"
            echo "  --no-routing           Disable dynamic prompt routing (UPGRADE-012)"
            echo "  --duration HOURS       Max session hours (default: 10)"
            echo "  --ric-mode MODE        RIC Loop enforcement: ENFORCED | SUGGESTED | DISABLED"
            echo "  --ric-init UPGRADE-ID  Auto-initialize RIC session with given upgrade ID"
            echo "  --help                 Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 'Implement feature X'"
            echo "  $0 --model opus --with-recovery 'Complex refactoring'"
            echo "  $0 --continuous --with-recovery 'Long overnight session'"
            echo "  $0 --ric-mode ENFORCED --ric-init UPGRADE-016 'RIC Loop task'"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            TASK_DESC_ARG="$1"
            shift
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."

    # Check tmux is installed
    if ! command -v tmux &> /dev/null; then
        log_error "tmux not found. Install with: sudo apt install tmux"
        exit 1
    fi
    log_success "tmux available"

    # Check Claude Code is installed
    if ! command -v claude &> /dev/null; then
        log_error "Claude Code CLI not found. Install from https://claude.ai"
        exit 1
    fi
    log_success "Claude Code CLI found"

    # Check Claude authentication
    if ! claude whoami &> /dev/null; then
        log_error "Not authenticated. Run 'claude login' first"
        exit 1
    fi
    log_success "Claude Code authenticated"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found"
        exit 1
    fi
    log_success "Python 3 found"

    # Check virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        log_success "Virtual environment active: $(basename $VIRTUAL_ENV)"
    else
        log_warn "No virtual environment active - dependencies may be missing"
    fi

    # Check psutil
    if ! python3 -c "import psutil" &> /dev/null; then
        log_warn "psutil not installed. Installing..."
        pip install psutil 2>/dev/null || python3 -m pip install psutil 2>/dev/null || log_error "Could not install psutil"
    fi
    log_success "psutil available"

    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        log_success "Docker available (sandbox mode possible)"
    else
        log_warn "Docker not available (sandbox mode disabled)"
    fi

    # Check environment variables
    if [ -z "$QC_USER_ID" ] || [ -z "$QC_API_TOKEN" ]; then
        log_warn "QuantConnect credentials not set (QC_USER_ID, QC_API_TOKEN)"
    else
        log_success "QuantConnect credentials configured"
    fi

    # Create directories
    mkdir -p "$LOGS_DIR"
    mkdir -p "$PROJECT_DIR/config"
    log_success "Directories ready"

    log_info "Pre-flight checks complete"
}

# Initialize progress file
init_progress_file() {
    local task_desc="${1:-Autonomous development session}"

    log_info "Initializing progress file..."

    cat > "$PROGRESS_FILE" << EOF
# Session: $SESSION_ID
# Started: $(date -Iseconds)
# Goal: $task_desc

## Session Configuration
- Max Runtime: 10 hours
- Checkpoint Interval: 15 minutes
- Watchdog: Active

## Completed
(none yet)

## In Progress
- [ ] Initial setup and environment check

## Next Steps
- [ ] Review existing code state
- [ ] Plan implementation approach
- [ ] Execute planned tasks
- [ ] Run tests and validate
- [ ] Commit checkpoint

## Blockers
None

## Notes
Session started at $(date)
EOF

    log_success "Progress file initialized: $PROGRESS_FILE"
}

# Initialize session notes file (relay-race pattern for context persistence)
init_session_notes() {
    local task_desc="${1:-Autonomous development session}"
    local notes_file="$PROJECT_DIR/claude-session-notes.md"
    local template_file="$SCRIPT_DIR/templates/session-notes-template.md"

    # Only create if doesn't exist (preserve notes across sessions for relay-race)
    if [ ! -f "$notes_file" ]; then
        if [ -f "$template_file" ]; then
            cp "$template_file" "$notes_file"
            # Update with session info
            sed -i "s/_timestamp_/$(date -Iseconds)/g" "$notes_file"
            sed -i "s/_session_id_/$SESSION_ID/g" "$notes_file"
            sed -i "s/_Replace with your current task\/goal_/$task_desc/g" "$notes_file"
            log_success "Session notes initialized: $notes_file"
        else
            # Create minimal notes file
            cat > "$notes_file" << NOTES_EOF
# Claude Session Notes

## Current Goal
$task_desc

## Key Decisions Made
_Document decisions here_

## Important Discoveries
_Document discoveries here_

## Next Steps
_Document next steps here_

---
**Last Updated**: $(date -Iseconds)
**Session ID**: $SESSION_ID
NOTES_EOF
            log_success "Session notes created: $notes_file"
        fi
    else
        log_info "Session notes exist (preserving relay context): $notes_file"
        # Update the session ID at bottom
        if grep -q "Session ID:" "$notes_file"; then
            sed -i "s/\*\*Session ID\*\*:.*/\*\*Session ID\*\*: $SESSION_ID/g" "$notes_file"
        fi
    fi
}

# Pre-load upgrade guide with domain knowledge
preload_upgrade_guide() {
    log_info "Pre-loading upgrade guide..."

    # Generate comprehensive upgrade guide
    if [ -f "$SCRIPT_DIR/preload_upgrade_guide.py" ]; then
        python3 "$SCRIPT_DIR/preload_upgrade_guide.py" \
            --progress-file "$PROGRESS_FILE" \
            --output "$PROJECT_DIR/claude-upgrade-guide.md" \
            2>/dev/null && log_success "Upgrade guide generated: claude-upgrade-guide.md" \
            || log_warn "Could not generate upgrade guide"
    fi
}

# Run proactive issue detection
run_issue_detection() {
    log_info "Running proactive issue detection..."

    if [ -f "$SCRIPT_DIR/issue_detector.py" ]; then
        ISSUE_REPORT="$LOGS_DIR/issues-$SESSION_ID.md"
        python3 "$SCRIPT_DIR/issue_detector.py" \
            --scan \
            --report "$ISSUE_REPORT" \
            --project "$PROJECT_DIR" \
            2>/dev/null

        if [ -f "$ISSUE_REPORT" ]; then
            # Count issues by priority
            P0_COUNT=$(grep -c "^### ISSUE.*P0" "$ISSUE_REPORT" 2>/dev/null || echo "0")
            P1_COUNT=$(grep -c "^### ISSUE.*P1" "$ISSUE_REPORT" 2>/dev/null || echo "0")
            P2_COUNT=$(grep -c "^### ISSUE.*P2" "$ISSUE_REPORT" 2>/dev/null || echo "0")

            log_success "Issue detection complete: P0=$P0_COUNT, P1=$P1_COUNT, P2=$P2_COUNT"

            if [ "$P0_COUNT" -gt 0 ]; then
                log_warn "Found $P0_COUNT P0 (critical) issues - address these first!"
            fi
        fi
    else
        log_warn "Issue detector not found, skipping"
    fi
}

# Create watchdog config if not exists
init_watchdog_config() {
    if [ ! -f "$WATCHDOG_CONFIG" ]; then
        log_info "Creating watchdog configuration..."

        cat > "$WATCHDOG_CONFIG" << EOF
{
    "max_runtime_hours": 10,
    "max_idle_minutes": 30,
    "max_cost_usd": 50.0,
    "checkpoint_interval_minutes": 15,
    "log_file": "logs/watchdog.log",
    "progress_file": "claude-progress.txt",
    "budget_file": "logs/budget.json",
    "check_interval_seconds": 30,
    "alert_email": null,
    "slack_webhook": null
}
EOF
        log_success "Watchdog config created: $WATCHDOG_CONFIG"
    else
        log_success "Using existing watchdog config"
    fi
}

# Start watchdog
start_watchdog() {
    if [ "$ENABLE_WATCHDOG" != "true" ]; then
        log_warn "Watchdog disabled (--no-watchdog)"
        return 0
    fi

    log_info "Starting watchdog process..."

    # Kill any existing watchdog
    pkill -f "watchdog.py" 2>/dev/null || true

    # Start watchdog in background
    python3 "$SCRIPT_DIR/watchdog.py" \
        --config "$WATCHDOG_CONFIG" \
        >> "$LOGS_DIR/watchdog-$SESSION_ID.log" 2>&1 &

    WATCHDOG_PID=$!
    echo $WATCHDOG_PID > "$LOGS_DIR/watchdog.pid"

    # Wait a moment and check it's running
    sleep 2
    if ps -p $WATCHDOG_PID > /dev/null; then
        log_success "Watchdog started (PID: $WATCHDOG_PID)"
    else
        log_error "Watchdog failed to start"
        exit 1
    fi
}

# Start auto-resume monitor
start_auto_resume() {
    if [ "$WITH_RECOVERY" != "true" ]; then
        return 0
    fi

    log_info "Starting auto-resume monitor..."

    # Kill any existing auto-resume
    pkill -f "auto-resume.sh" 2>/dev/null || true

    # Start auto-resume in background
    AUTO_RESUME_MAX_RESTARTS=5 \
    AUTO_RESUME_BACKOFF_BASE=30 \
    "$SCRIPT_DIR/auto-resume.sh" \
        >> "$LOGS_DIR/auto-resume-$SESSION_ID.log" 2>&1 &

    AUTO_RESUME_PID=$!
    echo $AUTO_RESUME_PID > "$LOGS_DIR/auto-resume.pid"

    sleep 1
    if ps -p $AUTO_RESUME_PID > /dev/null 2>&1; then
        log_success "Auto-resume monitor started (PID: $AUTO_RESUME_PID)"
    else
        log_warn "Auto-resume monitor failed to start (continuing without)"
    fi
}

# Send session start notification
send_start_notification() {
    local task_desc="$1"

    if [ -f "$SCRIPT_DIR/notify.py" ]; then
        python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
try:
    from notify import notify_session_start
    notify_session_start(
        task='$task_desc',
        max_hours=$MAX_DURATION_HOURS,
        model='$MODEL'
    )
except Exception as e:
    print(f'Notification failed: {e}', file=sys.stderr)
" 2>/dev/null || true
    fi
}

# Start Claude Code session
start_claude_session() {
    local task_desc="${1:-Continue development work. Check claude-progress.txt for current state.}"

    log_info "Starting Claude Code session..."
    log_info "Task: $task_desc"

    # Create session log
    SESSION_LOG="$LOGS_DIR/claude-$SESSION_ID.log"

    # UPGRADE-012: Use hierarchical prompt system if enabled
    DYNAMIC_PROMPT=""
    ROUTING_DECISION=""
    if [ "$USE_DYNAMIC_ROUTING" = "true" ]; then
        log_info "Using dynamic prompt routing (UPGRADE-012)..."

        # Call the task router
        if ROUTING_DECISION=$(python3 "$SCRIPT_DIR/select_prompts.py" --json "$task_desc" 2>/dev/null); then
            COMPLEXITY=$(echo "$ROUTING_DECISION" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('complexity','unknown'))" 2>/dev/null || echo "unknown")
            DOMAIN=$(echo "$ROUTING_DECISION" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('domain','general'))" 2>/dev/null || echo "general")
            SCORE=$(echo "$ROUTING_DECISION" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('complexity_score',0))" 2>/dev/null || echo "0")

            log_success "Task routed: complexity=$COMPLEXITY, domain=$DOMAIN, score=$SCORE"

            # Get the assembled prompt from the router
            DYNAMIC_PROMPT=$(python3 "$SCRIPT_DIR/select_prompts.py" "$task_desc" 2>/dev/null)

            if [ -n "$DYNAMIC_PROMPT" ]; then
                log_success "Dynamic prompt generated successfully"
            else
                log_warn "Dynamic prompt empty, falling back to default"
            fi
        else
            log_warn "Task router failed, falling back to default prompt"
        fi
    fi

    # Build prompt - use dynamic if available, otherwise static default
    if [ -n "$DYNAMIC_PROMPT" ]; then
        # Prepend session context to the dynamic prompt
        read -r -d '' SESSION_CONTEXT << CONTEXT_EOF || true
You are starting an autonomous development session. Session ID: $SESSION_ID

=== CRITICAL: COMPLETE ALL TASKS ===
This overnight session must complete ALL tasks (P0, P1, AND P2).
- P0 tasks are critical
- P1 tasks are important (REQUIRED)
- P2 tasks are nice-to-have (STILL REQUIRED for this session)

DO NOT STOP after completing just P0 categories.
The stop hook will block you from stopping until ALL tasks are complete.

RELAY-RACE PATTERN (Context Persistence):
- claude-session-notes.md persists across sessions - use it to pass context
- Document important discoveries and decisions there
- Update "Next Steps" section before session ends

TODO LIST SYNC (CRITICAL):
- The TodoWrite tool updates your internal todo list
- Copy pending tasks to "Task Status" section in claude-session-notes.md
- When completing a task: mark [x] in progress file AND add to "Completed This Session"
- Keep session notes in sync with progress file for crash recovery

COMPLETION REQUIREMENTS:
- ALL P0 categories must be complete
- ALL P1 categories must be complete
- ALL P2 categories must be complete
- Stop hook enforces 100% completion by default

ADDITIONAL CONTEXT FILES:
- claude-upgrade-guide.md: Comprehensive implementation guide with domain knowledge
- logs/issues-*.md: Proactive issue detection report (address P0 issues first)

Start by:
1. Reading claude-progress.txt for task list
2. Reading claude-upgrade-guide.md for implementation guidance
3. Reading claude-session-notes.md for context
4. Reading claude-recovery-context.md if exists (compaction recovery)
5. Then begin working on the NEXT pending task (any priority)

CONTEXT_EOF
        INITIAL_PROMPT="$SESSION_CONTEXT

---
$DYNAMIC_PROMPT"
    else
        # Fall back to static prompt
        read -r -d '' INITIAL_PROMPT << EOF || true
You are starting an autonomous development session. Session ID: $SESSION_ID

=== CRITICAL: COMPLETE ALL TASKS ===
This overnight session must complete ALL tasks (P0, P1, AND P2).
- P0 tasks are critical
- P1 tasks are important (REQUIRED)
- P2 tasks are nice-to-have (STILL REQUIRED for this session)

DO NOT STOP after completing just P0 categories.
The stop hook will block you from stopping until ALL tasks are complete.

IMPORTANT INSTRUCTIONS:
1. Read claude-progress.txt to understand current state and tasks
2. Read claude-session-notes.md for context from previous sessions (relay-race pattern)
3. Read claude-recovery-context.md if it exists (post-compaction recovery)
4. Update progress file as you complete tasks (mark items with [x])
5. Update session notes with key decisions, discoveries, and context for future sessions
6. Commit checkpoints every 15-20 minutes
7. If you encounter blockers, document them and try alternative approaches
8. Focus on the goal: $task_desc

RELAY-RACE PATTERN (Context Persistence):
- claude-session-notes.md persists across sessions - use it to pass context
- Document important discoveries and decisions there
- Update "Next Steps" section before session ends

TODO LIST SYNC (CRITICAL):
- The TodoWrite tool updates your internal todo list
- Copy pending tasks to "Task Status" section in claude-session-notes.md
- When completing a task: mark [x] in progress file AND add to "Completed This Session"
- Keep session notes in sync with progress file for crash recovery
- This ensures context survives session restarts

COMPLETION REQUIREMENTS:
- ALL P0 categories must be complete
- ALL P1 categories must be complete
- ALL P2 categories must be complete
- Stop hook enforces 100% completion by default

ADDITIONAL CONTEXT FILES:
- claude-upgrade-guide.md: Comprehensive implementation guide with domain knowledge
- logs/issues-*.md: Proactive issue detection report (address P0 issues first)

Start by:
1. Reading claude-progress.txt for task list
2. Reading claude-upgrade-guide.md for implementation guidance
3. Reading claude-session-notes.md for context
4. Reading claude-recovery-context.md if exists (compaction recovery)
5. Copying any pending tasks to session notes "Task Status" section
6. Then begin working on the NEXT pending task (any priority)
EOF
    fi

    echo "$INITIAL_PROMPT"
    echo ""
    echo "=============================================="
    echo "Session starting. Press Ctrl+C to stop."
    echo "Logs: $SESSION_LOG"
    echo "Progress: $PROGRESS_FILE"
    echo "=============================================="
    echo ""

    # Start Claude Code
    # Use tee to log while also showing output
    cd "$PROJECT_DIR"

    # Export CONTINUOUS_MODE for Stop hook if enabled
    if [ "$CONTINUOUS_MODE" = "true" ]; then
        export CONTINUOUS_MODE=1
        log_info "Continuous mode enabled - Claude will continue until all tasks complete"
    fi

    # UPGRADE-016: Export RIC_MODE for enforcement hooks
    export RIC_MODE="$RIC_MODE"
    log_info "RIC Loop mode: $RIC_MODE"

    # UPGRADE-017/v4.3: Enable autonomous mode for overnight sessions
    # This prevents blocking on doc violations - logs issues for later review instead
    export RIC_AUTONOMOUS_MODE=1
    export RIC_ENFORCEMENT_LEVEL="WARN"  # Warn but don't block
    log_info "Autonomous mode enabled - doc enforcement will warn, not block"

    # Auto-initialize RIC session if requested
    if [ -n "$AUTO_RIC_INIT" ]; then
        log_info "Auto-initializing RIC Loop session: $AUTO_RIC_INIT"
        if python3 "$PROJECT_DIR/.claude/hooks/ric_state_manager.py" init \
            --upgrade-id "$AUTO_RIC_INIT" \
            --title "$task_desc" 2>/dev/null; then
            log_success "RIC Loop session initialized: $AUTO_RIC_INIT"

            # Add RIC instructions to prompt
            RIC_CONTEXT="
=== RIC LOOP SESSION ACTIVE ===
This session uses the Meta-RIC Loop v3.0 methodology.
Upgrade ID: $AUTO_RIC_INIT
Mode: $RIC_MODE

CRITICAL RIC RULES:
1. Start at Phase 0 (Research) - do online research FIRST
2. Log every phase: [ITERATION X/5] === PHASE N: NAME ===
3. Minimum 3 iterations before exit allowed
4. Classify insights as P0/P1/P2 - ALL must be resolved
5. Single-component commits in Phase 3

Check state: python3 .claude/hooks/ric_state_manager.py status
Advance phase: python3 .claude/hooks/ric_state_manager.py advance
Add insight: python3 .claude/hooks/ric_state_manager.py add-insight --priority P0 --description \"...\"

Start with: [ITERATION 1/5] === PHASE 0: RESEARCH ===
"
            INITIAL_PROMPT="$RIC_CONTEXT

$INITIAL_PROMPT"
        else
            log_warn "Failed to initialize RIC session (continuing without)"
        fi
    fi

    claude --print "$INITIAL_PROMPT" 2>&1 | tee -a "$SESSION_LOG"
}

# Cleanup on exit
cleanup() {
    log_info "Cleaning up..."

    # Stop watchdog
    if [ -f "$LOGS_DIR/watchdog.pid" ]; then
        WATCHDOG_PID=$(cat "$LOGS_DIR/watchdog.pid")
        if ps -p $WATCHDOG_PID > /dev/null 2>&1; then
            kill $WATCHDOG_PID 2>/dev/null || true
            log_info "Watchdog stopped"
        fi
        rm -f "$LOGS_DIR/watchdog.pid"
    fi

    # Stop auto-resume monitor
    if [ -f "$LOGS_DIR/auto-resume.pid" ]; then
        AUTO_RESUME_PID=$(cat "$LOGS_DIR/auto-resume.pid")
        if ps -p $AUTO_RESUME_PID > /dev/null 2>&1; then
            kill $AUTO_RESUME_PID 2>/dev/null || true
            log_info "Auto-resume monitor stopped"
        fi
        rm -f "$LOGS_DIR/auto-resume.pid"
    fi

    # Update progress file
    echo "" >> "$PROGRESS_FILE"
    echo "## Session End" >> "$PROGRESS_FILE"
    echo "- Ended: $(date -Iseconds)" >> "$PROGRESS_FILE"
    echo "- Model: $MODEL" >> "$PROGRESS_FILE"
    echo "- Recovery enabled: $WITH_RECOVERY" >> "$PROGRESS_FILE"

    # UPGRADE-012.1: Log session outcome
    if [ "$USE_DYNAMIC_ROUTING" = "true" ] && [ -n "$ROUTING_DECISION" ]; then
        log_info "Logging session outcome (UPGRADE-012.1)..."

        # Extract routing info
        COMPLEXITY=$(echo "$ROUTING_DECISION" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('complexity_level','L1_moderate'))" 2>/dev/null || echo "L1_moderate")
        COMPLEXITY_SCORE=$(echo "$ROUTING_DECISION" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('complexity_score',0))" 2>/dev/null || echo "0")
        DEPTH_SCORE=$(echo "$ROUTING_DECISION" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('depth_score',0.0))" 2>/dev/null || echo "0.0")
        WIDTH_SCORE=$(echo "$ROUTING_DECISION" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('width_score',0.0))" 2>/dev/null || echo "0.0")
        DOMAIN=$(echo "$ROUTING_DECISION" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('domain','general'))" 2>/dev/null || echo "general")

        # Count tasks from progress file
        TASKS_COMPLETED=$(grep -c '\[x\]' "$PROGRESS_FILE" 2>/dev/null || echo "0")
        TASKS_TOTAL=$(grep -c '\[ \]\|\[x\]' "$PROGRESS_FILE" 2>/dev/null || echo "0")

        # Determine status
        if [ "$TASKS_TOTAL" -eq 0 ]; then
            STATUS="partial"
        elif [ "$TASKS_COMPLETED" -eq "$TASKS_TOTAL" ]; then
            STATUS="success"
        elif [ "$TASKS_COMPLETED" -gt 0 ]; then
            STATUS="partial"
        else
            STATUS="failed"
        fi

        # Calculate duration
        SESSION_END=$(date +%s)
        SESSION_START_TS=$(date -d "${SESSION_ID:0:8} ${SESSION_ID:9:2}:${SESSION_ID:11:2}:${SESSION_ID:13:2}" +%s 2>/dev/null || echo "$SESSION_END")
        DURATION_MINS=$(( (SESSION_END - SESSION_START_TS) / 60 ))

        # Log the outcome
        python3 "$SCRIPT_DIR/log_session_outcome.py" \
            --session-id "$SESSION_ID" \
            --task "${TASK_DESC:-Unknown task}" \
            --complexity "$COMPLEXITY" \
            --complexity-score "$COMPLEXITY_SCORE" \
            --depth-score "$DEPTH_SCORE" \
            --width-score "$WIDTH_SCORE" \
            --domain "$DOMAIN" \
            --status "$STATUS" \
            --tasks-completed "$TASKS_COMPLETED" \
            --tasks-total "$TASKS_TOTAL" \
            --duration "$DURATION_MINS" \
            2>/dev/null && log_success "Session outcome logged" || log_warn "Failed to log session outcome"

        # Update session notes with trends
        SESSION_NOTES="$PROJECT_DIR/claude-session-notes.md"
        if [ -f "$SESSION_NOTES" ]; then
            TRENDS_SUMMARY=$(python3 "$SCRIPT_DIR/analyze_session_outcomes.py" --session-notes 2>/dev/null)
            if [ -n "$TRENDS_SUMMARY" ]; then
                # Append trends to session notes
                {
                    echo ""
                    echo "---"
                    echo ""
                    echo "$TRENDS_SUMMARY"
                    echo ""
                    echo "_Updated: $(date -Iseconds)_"
                } >> "$SESSION_NOTES"
                log_success "Session notes updated with outcome trends"
            fi
        fi

        # UPGRADE-012.2: Run ACE Reflector for pattern analysis
        log_info "Running ACE Reflector pattern analysis (UPGRADE-012.2)..."
        REFLECTOR_OUTPUT="$LOGS_DIR/ace-reflector-$(date +%Y%m%d).json"
        if python3 "$SCRIPT_DIR/ace_reflector.py" --analyze --days 7 --output "$REFLECTOR_OUTPUT" 2>/dev/null; then
            log_success "ACE Reflector analysis saved to $REFLECTOR_OUTPUT"
            # Log any high-confidence recommendations
            if [ -f "$REFLECTOR_OUTPUT" ]; then
                HIGH_CONF=$(python3 -c "
import json
with open('$REFLECTOR_OUTPUT') as f:
    data = json.load(f)
patterns = data.get('patterns_found', [])
high_conf = [p for p in patterns if p.get('confidence', 0) >= 0.7]
if high_conf:
    print(f'{len(high_conf)} high-confidence patterns found')
" 2>/dev/null)
                if [ -n "$HIGH_CONF" ]; then
                    log_info "ACE Reflector: $HIGH_CONF"
                fi
            fi
        else
            log_warn "ACE Reflector analysis failed or no sessions to analyze"
        fi
    fi

    log_info "Session ended"
}

# Set up trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    echo ""
    echo "=============================================="
    echo "  Autonomous Claude Code Session Launcher"
    echo "  Session ID: $SESSION_ID"
    echo "=============================================="
    echo ""
    echo "Configuration:"
    echo "  Model: $MODEL"
    echo "  Max Duration: ${MAX_DURATION_HOURS} hours"
    echo "  Watchdog: $ENABLE_WATCHDOG"
    echo "  Auto-Recovery: $WITH_RECOVERY"
    echo "  Continuous Mode: $CONTINUOUS_MODE"
    echo "  Dynamic Routing: $USE_DYNAMIC_ROUTING"
    echo "  RIC Mode: $RIC_MODE"
    if [ -n "$AUTO_RIC_INIT" ]; then
        echo "  RIC Init: $AUTO_RIC_INIT"
    fi
    echo ""

    # Get task description from argument or prompt
    TASK_DESC="${TASK_DESC_ARG:-}"
    if [ -z "$TASK_DESC" ]; then
        echo "Enter task description (or press Enter for default):"
        read -r TASK_DESC
        TASK_DESC="${TASK_DESC:-Autonomous development session}"
    fi

    # Run setup
    preflight_checks
    init_watchdog_config
    init_progress_file "$TASK_DESC"
    init_session_notes "$TASK_DESC"

    # NEW: Pre-load context and detect issues
    preload_upgrade_guide
    run_issue_detection

    start_watchdog
    start_auto_resume
    send_start_notification "$TASK_DESC"

    # Start session
    start_claude_session "$TASK_DESC"
}

# Only run main when executed directly, not when sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi
