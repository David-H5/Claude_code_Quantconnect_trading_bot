#!/bin/bash
#
# WSL2 Persistence Setup for Overnight Claude Code Sessions
#
# This script configures WSL2 for persistent overnight sessions:
# 1. Creates recommended .wslconfig on Windows host
# 2. Creates recommended wsl.conf in Ubuntu
# 3. Installs tmux with recommended config
# 4. Verifies the setup
#
# Usage:
#   ./scripts/setup_wsl_overnight.sh
#
# After running, execute: wsl --shutdown from PowerShell, then reopen Ubuntu
#

set -e

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

echo ""
echo "=============================================="
echo "  WSL2 Persistence Setup for Overnight Sessions"
echo "=============================================="
echo ""

# Detect if running in WSL
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    log_error "This script should be run inside WSL2"
    exit 1
fi

log_success "Running in WSL2"

# Get Windows username for .wslconfig path
WIN_USER=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r\n' || echo "")
if [ -z "$WIN_USER" ]; then
    log_warn "Could not detect Windows username"
    echo "Please enter your Windows username:"
    read -r WIN_USER
fi

WIN_HOME="/mnt/c/Users/$WIN_USER"
if [ ! -d "$WIN_HOME" ]; then
    log_error "Windows home not found at: $WIN_HOME"
    echo "Please check your Windows username"
    exit 1
fi

log_success "Windows home: $WIN_HOME"

# =============================================================================
# Step 1: Create .wslconfig on Windows host
# =============================================================================
echo ""
log_info "Step 1: Creating .wslconfig on Windows host"

WSLCONFIG="$WIN_HOME/.wslconfig"
WSLCONFIG_CONTENT='[wsl2]
# Memory allocation for WSL2 (adjust based on your system)
memory=8GB

# Number of processors
processors=4

# Swap file size
swap=4GB

# Enable localhost forwarding
localhostForwarding=true

[experimental]
# Use sparse VHD to save disk space
sparseVhd=true
'

if [ -f "$WSLCONFIG" ]; then
    log_warn ".wslconfig already exists"
    echo "Current contents:"
    cat "$WSLCONFIG" | sed 's/^/  /'
    echo ""
    echo "Do you want to overwrite it? (y/N)"
    read -r OVERWRITE
    if [ "$OVERWRITE" != "y" ] && [ "$OVERWRITE" != "Y" ]; then
        log_info "Keeping existing .wslconfig"
    else
        echo "$WSLCONFIG_CONTENT" > "$WSLCONFIG"
        log_success "Created .wslconfig"
    fi
else
    echo "$WSLCONFIG_CONTENT" > "$WSLCONFIG"
    log_success "Created .wslconfig at $WSLCONFIG"
fi

# =============================================================================
# Step 2: Create wsl.conf in Ubuntu
# =============================================================================
echo ""
log_info "Step 2: Creating /etc/wsl.conf"

WSLCONF_CONTENT="[boot]
# Enable systemd for background services
systemd=true
# Start cron for scheduled tasks
command=\"service cron start\"

[automount]
# Enable Windows drive mounting
enabled=true
# Set proper file permissions
options=\"metadata,umask=22,fmask=11\"

[user]
# Set default user (change to your username)
default=$(whoami)

[interop]
# Enable running Windows executables
enabled=true
appendWindowsPath=true
"

if [ -f "/etc/wsl.conf" ]; then
    log_warn "/etc/wsl.conf already exists"
    echo "Current contents:"
    cat /etc/wsl.conf | sed 's/^/  /'
    echo ""
    echo "Do you want to overwrite it? (requires sudo) (y/N)"
    read -r OVERWRITE
    if [ "$OVERWRITE" != "y" ] && [ "$OVERWRITE" != "Y" ]; then
        log_info "Keeping existing wsl.conf"
    else
        echo "$WSLCONF_CONTENT" | sudo tee /etc/wsl.conf > /dev/null
        log_success "Created /etc/wsl.conf"
    fi
else
    echo "$WSLCONF_CONTENT" | sudo tee /etc/wsl.conf > /dev/null
    log_success "Created /etc/wsl.conf"
fi

# =============================================================================
# Step 3: Install and configure tmux
# =============================================================================
echo ""
log_info "Step 3: Installing and configuring tmux"

if ! command -v tmux &> /dev/null; then
    log_info "Installing tmux..."
    sudo apt update && sudo apt install -y tmux
    log_success "tmux installed"
else
    log_success "tmux already installed: $(tmux -V)"
fi

TMUX_CONF="$HOME/.tmux.conf"
TMUX_CONF_CONTENT='# Tmux configuration for overnight Claude Code sessions

# Increase scrollback buffer
set -g history-limit 50000

# Enable mouse support
set -g mouse on

# Reduce escape time for vim
set -sg escape-time 0

# Use vi key bindings
setw -g mode-keys vi

# Status bar configuration
set -g status on
set -g status-interval 5
set -g status-position bottom
set -g status-left "[#S] "
set -g status-right "#{pane_current_path} | %H:%M"
set -g status-style "bg=black,fg=white"

# Easy pane splitting
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"

# Prevent automatic window renaming
set-option -g allow-rename off

# Activity monitoring
setw -g monitor-activity on
set -g visual-activity on

# Start windows and panes at 1, not 0
set -g base-index 1
setw -g pane-base-index 1

# Renumber windows when one is closed
set -g renumber-windows on

# Enable 256 colors
set -g default-terminal "screen-256color"

# Quick reload config
bind r source-file ~/.tmux.conf \; display "Config reloaded!"
'

if [ -f "$TMUX_CONF" ]; then
    log_warn ".tmux.conf already exists"
    echo "Do you want to overwrite it? (y/N)"
    read -r OVERWRITE
    if [ "$OVERWRITE" != "y" ] && [ "$OVERWRITE" != "Y" ]; then
        log_info "Keeping existing .tmux.conf"
    else
        echo "$TMUX_CONF_CONTENT" > "$TMUX_CONF"
        log_success "Created .tmux.conf"
    fi
else
    echo "$TMUX_CONF_CONTENT" > "$TMUX_CONF"
    log_success "Created .tmux.conf at $TMUX_CONF"
fi

# =============================================================================
# Step 4: Install additional dependencies
# =============================================================================
echo ""
log_info "Step 4: Installing additional dependencies"

# Install jq for JSON parsing
if ! command -v jq &> /dev/null; then
    log_info "Installing jq..."
    sudo apt install -y jq
    log_success "jq installed"
else
    log_success "jq already installed"
fi

# Install bc for calculations
if ! command -v bc &> /dev/null; then
    log_info "Installing bc..."
    sudo apt install -y bc
    log_success "bc installed"
else
    log_success "bc already installed"
fi

# Install psutil for Python
if ! python3 -c "import psutil" 2>/dev/null; then
    log_info "Installing psutil..."
    pip3 install psutil
    log_success "psutil installed"
else
    log_success "psutil already installed"
fi

# =============================================================================
# Step 5: Verification
# =============================================================================
echo ""
log_info "Step 5: Verification"

CHECKS_PASSED=0
CHECKS_TOTAL=5

# Check .wslconfig
if [ -f "$WSLCONFIG" ]; then
    log_success ".wslconfig exists"
    ((CHECKS_PASSED++))
else
    log_warn ".wslconfig not found"
fi

# Check wsl.conf
if [ -f "/etc/wsl.conf" ]; then
    log_success "wsl.conf exists"
    ((CHECKS_PASSED++))
else
    log_warn "wsl.conf not found"
fi

# Check tmux
if command -v tmux &> /dev/null; then
    log_success "tmux available"
    ((CHECKS_PASSED++))
else
    log_warn "tmux not available"
fi

# Check .tmux.conf
if [ -f "$TMUX_CONF" ]; then
    log_success ".tmux.conf exists"
    ((CHECKS_PASSED++))
else
    log_warn ".tmux.conf not found"
fi

# Check systemd
if pidof systemd &> /dev/null; then
    log_success "systemd is running"
    ((CHECKS_PASSED++))
else
    log_warn "systemd not running (restart WSL after setup)"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "  Setup Complete: $CHECKS_PASSED/$CHECKS_TOTAL checks passed"
echo "=============================================="
echo ""

if [ "$CHECKS_PASSED" -eq "$CHECKS_TOTAL" ]; then
    log_success "All checks passed!"
else
    log_warn "Some checks require WSL restart"
fi

echo ""
echo "IMPORTANT: To apply changes, run from PowerShell:"
echo ""
echo "  wsl --shutdown"
echo ""
echo "Then reopen your Ubuntu terminal."
echo ""
echo "Tmux Quick Reference:"
echo "  tmux new -s overnight     Create named session"
echo "  tmux attach -t overnight  Attach to session"
echo "  Ctrl+b d                  Detach from session"
echo "  Ctrl+b [                  Scroll mode (q to exit)"
echo ""
echo "Start overnight session:"
echo "  ./scripts/run_overnight.sh \"Your task\""
echo ""
