#!/bin/bash
#
# Cron Scheduling Setup for Overnight Claude Code Sessions
#
# This script helps set up cron jobs for automated overnight sessions.
#
# Usage:
#   ./scripts/setup_cron.sh              - Show current cron config
#   ./scripts/setup_cron.sh add          - Add overnight session cron
#   ./scripts/setup_cron.sh remove       - Remove overnight session cron
#   ./scripts/setup_cron.sh examples     - Show example cron configurations
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CRON_MARKER="# CLAUDE_OVERNIGHT_SESSION"

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

show_current() {
    echo "Current Claude overnight cron jobs:"
    echo "===================================="
    crontab -l 2>/dev/null | grep -A1 "$CRON_MARKER" || echo "(none configured)"
    echo ""
}

show_examples() {
    cat << 'EOF'
Example Cron Configurations
===========================

# Run overnight at 11 PM on weekdays
0 23 * * 1-5 cd /path/to/project && ./scripts/run_overnight.sh "Continue development"

# Run every night at midnight
0 0 * * * cd /path/to/project && ./scripts/run_overnight.sh --with-recovery "Nightly session"

# Run overnight with Opus model for complex work (weekends)
0 22 * * 6-7 cd /path/to/project && ./scripts/run_overnight.sh --model opus "Weekend deep work"

# Periodic checkpoints every 30 minutes during session hours
*/30 23-7 * * * cd /path/to/project && ./scripts/checkpoint.sh periodic 30

# Morning recovery check at 6 AM
0 6 * * * cd /path/to/project && ./scripts/checkpoint.sh recover

Cron Time Format
================
┌─────────── minute (0-59)
│ ┌───────── hour (0-23)
│ │ ┌─────── day of month (1-31)
│ │ │ ┌───── month (1-12)
│ │ │ │ ┌─── day of week (0-7, 0 or 7 = Sunday)
│ │ │ │ │
* * * * * command

Useful Time Patterns
====================
0 23 * * *     - Every day at 11 PM
0 23 * * 1-5   - Weekdays at 11 PM
0 22 * * 6-7   - Weekends at 10 PM
*/30 * * * *   - Every 30 minutes
0 */4 * * *    - Every 4 hours

To edit crontab manually: crontab -e
To view current crontab: crontab -l
EOF
}

add_cron() {
    local hour="${1:-23}"
    local days="${2:-1-5}"  # Default: weekdays
    local task="${3:-Continue autonomous development}"

    echo "Adding overnight session cron job..."
    echo ""
    echo "Configuration:"
    echo "  Time: $hour:00"
    echo "  Days: $days (1=Mon, 7=Sun)"
    echo "  Task: $task"
    echo "  Project: $PROJECT_DIR"
    echo ""

    # Create cron entry
    local cron_entry="0 $hour * * $days cd $PROJECT_DIR && ./scripts/run_overnight.sh --with-recovery \"$task\" $CRON_MARKER"

    # Check if already exists
    if crontab -l 2>/dev/null | grep -q "$CRON_MARKER"; then
        log_warn "Overnight session cron already exists. Remove first with: $0 remove"
        show_current
        return 1
    fi

    # Add to crontab
    (crontab -l 2>/dev/null || true; echo "$cron_entry") | crontab -

    log_success "Cron job added"
    echo ""
    show_current

    # Add periodic checkpoint job
    echo ""
    echo "Would you also like to add periodic checkpoint cron? (y/n)"
    read -r answer
    if [ "$answer" = "y" ]; then
        local checkpoint_entry="*/30 $hour-7 * * $days cd $PROJECT_DIR && ./scripts/checkpoint.sh periodic 30 $CRON_MARKER"
        (crontab -l 2>/dev/null || true; echo "$checkpoint_entry") | crontab -
        log_success "Periodic checkpoint cron added (every 30 min during session)"
    fi
}

remove_cron() {
    echo "Removing overnight session cron jobs..."

    if ! crontab -l 2>/dev/null | grep -q "$CRON_MARKER"; then
        log_warn "No overnight session cron jobs found"
        return 0
    fi

    # Remove lines with marker
    crontab -l 2>/dev/null | grep -v "$CRON_MARKER" | crontab - 2>/dev/null || true

    log_success "Overnight session cron jobs removed"
}

interactive_add() {
    echo "Interactive Cron Setup"
    echo "======================"
    echo ""

    # Get hour
    echo "What hour should the session start? (0-23, default: 23)"
    read -r hour
    hour="${hour:-23}"

    # Get days
    echo ""
    echo "Which days should it run?"
    echo "  1. Weekdays (Mon-Fri)"
    echo "  2. Weekends (Sat-Sun)"
    echo "  3. Every day"
    echo "  4. Custom (e.g., 1,3,5 for Mon,Wed,Fri)"
    read -r day_choice
    case "$day_choice" in
        1) days="1-5" ;;
        2) days="6-7" ;;
        3) days="*" ;;
        4)
            echo "Enter days (e.g., 1,3,5 or 1-5):"
            read -r days
            ;;
        *) days="1-5" ;;
    esac

    # Get task
    echo ""
    echo "Enter default task description (or press Enter for default):"
    read -r task
    task="${task:-Continue autonomous development}"

    echo ""
    add_cron "$hour" "$days" "$task"
}

# Main command router
case "${1:-}" in
    add)
        if [ -n "$2" ]; then
            add_cron "$2" "${3:-1-5}" "${4:-Continue autonomous development}"
        else
            interactive_add
        fi
        ;;
    remove)
        remove_cron
        ;;
    examples)
        show_examples
        ;;
    *)
        echo "Cron Scheduling for Overnight Sessions"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  add [hour] [days] [task]  Add overnight session cron"
        echo "  remove                    Remove overnight session cron"
        echo "  examples                  Show example configurations"
        echo ""
        echo "Examples:"
        echo "  $0 add              # Interactive setup"
        echo "  $0 add 23 1-5       # 11 PM on weekdays"
        echo "  $0 remove           # Remove all overnight crons"
        echo ""
        show_current
        ;;
esac
