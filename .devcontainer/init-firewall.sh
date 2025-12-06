#!/bin/bash
# init-firewall.sh - Network security for Claude Code devcontainer
# Implements default-deny policy with whitelisted domains
#
# This script runs on container creation to restrict outbound network access.
# Requires --cap-add=NET_ADMIN in devcontainer.json runArgs.

set -e

echo "Initializing firewall rules..."

# Check if we have iptables capability
if ! command -v iptables &> /dev/null; then
    echo "Warning: iptables not available, skipping firewall configuration"
    exit 0
fi

# Check if we have NET_ADMIN capability
if ! sudo iptables -L &> /dev/null 2>&1; then
    echo "Warning: No NET_ADMIN capability, skipping firewall configuration"
    exit 0
fi

# Flush existing rules
sudo iptables -F OUTPUT 2>/dev/null || true

# Allow loopback
sudo iptables -A OUTPUT -o lo -j ACCEPT

# Allow established connections
sudo iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow DNS (required for domain resolution)
sudo iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

# ============================================
# WHITELISTED HTTPS DESTINATIONS (port 443)
# ============================================

# Claude / Anthropic API
sudo iptables -A OUTPUT -p tcp --dport 443 -d api.anthropic.com -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport 443 -d console.anthropic.com -j ACCEPT

# QuantConnect API
sudo iptables -A OUTPUT -p tcp --dport 443 -d api.quantconnect.com -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport 443 -d www.quantconnect.com -j ACCEPT

# GitHub (for git operations)
sudo iptables -A OUTPUT -p tcp --dport 443 -d github.com -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport 443 -d api.github.com -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport 443 -d raw.githubusercontent.com -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport 443 -d objects.githubusercontent.com -j ACCEPT

# Git over SSH (optional, for SSH-based git operations)
sudo iptables -A OUTPUT -p tcp --dport 22 -d github.com -j ACCEPT

# Package registries
sudo iptables -A OUTPUT -p tcp --dport 443 -d registry.npmjs.org -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport 443 -d pypi.org -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport 443 -d files.pythonhosted.org -j ACCEPT

# Container registries (for devcontainer features)
sudo iptables -A OUTPUT -p tcp --dport 443 -d ghcr.io -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport 443 -d pkg-containers.githubusercontent.com -j ACCEPT

# ============================================
# OPTIONAL: LLM Provider APIs
# Uncomment as needed for your configuration
# ============================================

# OpenAI API (if using GPT models in ensemble)
# sudo iptables -A OUTPUT -p tcp --dport 443 -d api.openai.com -j ACCEPT

# Charles Schwab API (for brokerage integration)
# sudo iptables -A OUTPUT -p tcp --dport 443 -d api.schwab.com -j ACCEPT
# sudo iptables -A OUTPUT -p tcp --dport 443 -d api.schwabapi.com -j ACCEPT

# ============================================
# BLOCK EVERYTHING ELSE
# ============================================

# Drop all other HTTPS traffic
sudo iptables -A OUTPUT -p tcp --dport 443 -j DROP

# Drop HTTP (insecure)
sudo iptables -A OUTPUT -p tcp --dport 80 -j DROP

# Block VSCode IDE communication (prevents access to files outside container)
# Uncomment if you want complete isolation from IDE
# sudo iptables -A OUTPUT -p udp --dport 59778 -j DROP

echo "Firewall rules configured successfully."
echo ""
echo "Allowed destinations:"
echo "  - api.anthropic.com (Claude API)"
echo "  - api.quantconnect.com (QuantConnect)"
echo "  - github.com (Git operations)"
echo "  - registry.npmjs.org (NPM packages)"
echo "  - pypi.org (Python packages)"
echo "  - ghcr.io (Container registry)"
echo ""
echo "All other HTTPS traffic is blocked."

# Show current rules
echo ""
echo "Current OUTPUT rules:"
sudo iptables -L OUTPUT -n --line-numbers 2>/dev/null || echo "(Unable to list rules)"
