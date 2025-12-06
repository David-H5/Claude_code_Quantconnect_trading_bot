# Installation Guide

**Version**: 1.0.0 | **Last Updated**: November 2025

Step-by-step installation instructions for autonomous Claude Code development infrastructure.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Claude Code Setup](#claude-code-setup)
3. [Docker Sandbox Setup](#docker-sandbox-setup)
4. [Microsandbox Setup](#microsandbox-setup-self-hosted-microvms)
5. [Kubernetes Agent Sandbox](#kubernetes-agent-sandbox-enterprise)
6. [Watchdog Installation](#watchdog-installation)
7. [MCP Server Installation](#mcp-server-installation)
8. [Temporal.io Setup](#temporalio-setup)
9. [Budget Tracking Setup](#budget-tracking-setup)
10. [Verification](#verification)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10/11, macOS 12+, Ubuntu 20.04+ | Latest LTS |
| RAM | 8 GB | 16+ GB |
| Storage | 20 GB free | 50+ GB SSD |
| Docker | 24.0+ | 27.0+ |
| Python | 3.10+ | 3.11+ |
| Node.js | 18+ | 20+ LTS |

### Required Accounts

- [ ] Anthropic account with Claude Code access
- [ ] Claude Max plan (recommended: Max 20x for overnight sessions)
- [ ] QuantConnect account with API access
- [ ] GitHub account (for CI/CD)

### Optional Accounts (for MCP servers)

- [ ] Alpaca account (paper trading is free)
- [ ] Alpha Vantage API key (free tier available)
- [ ] Finviz Elite subscription ($40/month for MCP features)

---

## Claude Code Setup

### Step 1: Install Claude Code CLI

```bash
# macOS/Linux
curl -fsSL https://claude.ai/install.sh | sh

# Windows (PowerShell as Admin)
irm https://claude.ai/install.ps1 | iex

# Verify installation
claude --version
```

### Step 2: Authenticate

```bash
# Login to Claude
claude login

# Verify authentication
claude whoami
```

### Step 3: Configure Project Settings

Create `.claude/settings.json` in your project root:

```bash
# Create directory
mkdir -p .claude

# Copy example settings (if provided)
cp .claude/settings.example.json .claude/settings.json

# Or create from scratch
cat > .claude/settings.json << 'EOF'
{
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Bash(pytest:*)",
      "Bash(python:*)",
      "Bash(python3:*)",
      "Bash(pip:*)",
      "Bash(git status:*)",
      "Bash(git diff:*)",
      "Bash(git log:*)",
      "Bash(git add:*)",
      "Bash(lean:*)",
      "Bash(docker:*)"
    ],
    "ask": [
      "Write",
      "Edit",
      "Bash(git commit:*)",
      "Bash(git push:*)",
      "Bash(rm:*)"
    ]
  },
  "model": "claude-sonnet-4-5-20250929",
  "contextLimit": 0.6,
  "autoCompact": true
}
EOF
```

### Step 4: Install File Protection Hook

```bash
# Create hooks directory
mkdir -p .claude/hooks

# Create protection hook
cat > .claude/hooks/protect_files.py << 'PYTHON'
#!/usr/bin/env python3
"""
PreToolUse hook to protect sensitive files.
Exit code 2 blocks the operation.
"""
import sys
import json
import os

PROTECTED_PATTERNS = [
    ".env",
    "credentials",
    "secrets",
    ".claude/settings.json",
    "config/api_keys",
]

PROTECTED_EXACT = [
    ".env",
    ".env.local",
    ".env.production",
    "config/credentials.json",
]

def is_protected(file_path: str) -> bool:
    """Check if file path matches protected patterns."""
    normalized = file_path.replace("\\", "/").lower()

    # Check exact matches
    for protected in PROTECTED_EXACT:
        if normalized.endswith(protected.lower()):
            return True

    # Check patterns
    for pattern in PROTECTED_PATTERNS:
        if pattern.lower() in normalized:
            return True

    return False

def main():
    if len(sys.argv) < 2:
        sys.exit(0)

    try:
        tool_input = json.loads(sys.argv[1])
        file_path = tool_input.get("file_path", "")

        if file_path and is_protected(file_path):
            print(f"BLOCKED: Access to protected file: {file_path}")
            sys.exit(2)  # Exit code 2 blocks the operation
    except (json.JSONDecodeError, KeyError):
        pass

    sys.exit(0)

if __name__ == "__main__":
    main()
PYTHON

chmod +x .claude/hooks/protect_files.py
```

### Step 5: Configure Hooks in Settings

Update `.claude/settings.json` to include hooks:

```json
{
  "permissions": { ... },
  "hooks": {
    "PreToolUse": [{
      "matcher": "Edit|Write|Read",
      "hooks": [{
        "type": "command",
        "command": "python3 .claude/hooks/protect_files.py \"$TOOL_INPUT\""
      }]
    }]
  }
}
```

---

## Docker Sandbox Setup

### Step 1: Install Docker Desktop

```bash
# macOS
brew install --cask docker

# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Windows
# Download from https://www.docker.com/products/docker-desktop/
```

### Step 2: Enable Docker Sandboxes

Docker Desktop 4.50+ includes sandbox support:

```bash
# Verify Docker version
docker --version  # Should be 27.0+

# Check sandbox availability
docker sandbox --help
```

### Step 3: Configure Sandbox Settings

Create sandbox configuration:

```bash
# Create Docker sandbox config
mkdir -p ~/.docker

cat > ~/.docker/sandbox.json << 'EOF'
{
  "defaultMemory": "4g",
  "defaultCpus": 2,
  "networkPolicy": "restricted",
  "mountPolicy": "readonly-except-workspace",
  "persistWorkspace": true
}
EOF
```

### Step 4: Test Sandbox

```bash
# Run Claude Code in sandbox
docker sandbox run claude

# With workspace mount
docker sandbox run -w $(pwd) claude

# Verify isolation
docker sandbox run claude -c "echo 'Sandbox working'"
```

---

## Microsandbox Setup (Self-Hosted MicroVMs)

For production deployments requiring stronger isolation than Docker containers.

### Step 1: Check Platform Requirements

```bash
# Linux: Verify KVM support
ls -la /dev/kvm

# macOS: Verify Apple Silicon
uname -m  # Should be "arm64"
```

**Note**: macOS requires Apple Silicon (M1/M2/M3/M4). Intel Macs are not supported.

### Step 2: Install Microsandbox

```bash
# Option 1: Via Cargo (requires Rust)
cargo install microsandbox-server

# Option 2: Via install script
curl -fsSL https://get.microsandbox.dev | sh

# Verify installation
microsandbox --version
```

### Step 3: Start MCP Server

```bash
# Start with MCP integration
microsandbox serve --mcp

# Or run with specific configuration
microsandbox serve --mcp --port 8080 --max-sandboxes 10
```

### Step 4: Configure for Claude Code

Add to `.mcp.json`:

```json
{
  "mcpServers": {
    "microsandbox": {
      "command": "microsandbox",
      "args": ["serve", "--mcp"]
    }
  }
}
```

### Step 5: Test Isolation

```bash
# Create a test sandbox
microsandbox run --image python:3.11-slim \
  --memory 4096 \
  --cpus 2 \
  --network-policy deny-all \
  -- python3 -c "print('Hello from microVM')"
```

---

## Kubernetes Agent Sandbox (Enterprise)

For enterprise Kubernetes environments using the SIG Apps Agent Sandbox.

### Step 1: Install CRDs

```bash
# Apply Agent Sandbox CRDs
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/agent-sandbox/main/config/crd/bases/agent-sandbox.sigs.k8s.io_sandboxes.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/agent-sandbox/main/config/crd/bases/agent-sandbox.sigs.k8s.io_sandboxtemplates.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/agent-sandbox/main/config/crd/bases/agent-sandbox.sigs.k8s.io_sandboxclaims.yaml
```

### Step 2: Create SandboxTemplate

```yaml
# sandbox-template.yaml
apiVersion: agent-sandbox.sigs.k8s.io/v1alpha1
kind: SandboxTemplate
metadata:
  name: trading-bot-sandbox
  namespace: agents
spec:
  runtime: gvisor  # or kata-containers
  resources:
    limits:
      memory: "4Gi"
      cpu: "2"
    requests:
      memory: "2Gi"
      cpu: "1"
  warmPool:
    minReady: 3
    maxSize: 10
  securityContext:
    runAsNonRoot: true
    readOnlyRootFilesystem: true
  networkPolicy:
    egress:
      - to:
          - namespaceSelector:
              matchLabels:
                name: quantconnect
```

```bash
kubectl apply -f sandbox-template.yaml
```

### Step 3: Create Sandbox

```yaml
# sandbox.yaml
apiVersion: agent-sandbox.sigs.k8s.io/v1alpha1
kind: Sandbox
metadata:
  name: overnight-session-001
  namespace: agents
spec:
  templateRef:
    name: trading-bot-sandbox
  env:
    - name: SESSION_ID
      value: "overnight-session-001"
  volumeMounts:
    - name: workspace
      mountPath: /workspace
```

```bash
kubectl apply -f sandbox.yaml
```

### Step 4: Use SandboxClaim in Agent Frameworks

For LangChain, CrewAI, or other frameworks:

```python
from kubernetes import client, config

config.load_kube_config()
custom_api = client.CustomObjectsApi()

# Create SandboxClaim
sandbox_claim = {
    "apiVersion": "agent-sandbox.sigs.k8s.io/v1alpha1",
    "kind": "SandboxClaim",
    "metadata": {"name": "agent-session-001"},
    "spec": {
        "templateRef": {"name": "trading-bot-sandbox"},
        "ttl": "8h"
    }
}

custom_api.create_namespaced_custom_object(
    group="agent-sandbox.sigs.k8s.io",
    version="v1alpha1",
    namespace="agents",
    plural="sandboxclaims",
    body=sandbox_claim
)
```

---

## Watchdog Installation

### Step 1: Install Dependencies

```bash
# Install required packages
pip install psutil litellm python-dotenv

# Or add to requirements.txt
echo "psutil>=5.9.0" >> requirements.txt
echo "litellm>=1.0.0" >> requirements.txt
echo "python-dotenv>=1.0.0" >> requirements.txt
pip install -r requirements.txt
```

### Step 2: Create Watchdog Script

The watchdog script is provided at `scripts/watchdog.py`. Verify it exists:

```bash
ls -la scripts/watchdog.py
```

### Step 3: Configure Watchdog

Create watchdog configuration:

```bash
cat > config/watchdog.json << 'EOF'
{
  "max_runtime_hours": 10,
  "max_idle_minutes": 30,
  "max_cost_usd": 50.0,
  "checkpoint_interval_minutes": 15,
  "log_file": "logs/watchdog.log",
  "progress_file": "claude-progress.txt",
  "alert_email": null,
  "slack_webhook": null
}
EOF
```

### Step 4: Test Watchdog

```bash
# Run watchdog in test mode
python scripts/watchdog.py --test

# Run watchdog (background)
python scripts/watchdog.py &
```

---

## MCP Server Installation

### QuantConnect MCP (Primary)

```bash
# Pull Docker image
docker pull quantconnect/mcp-server:latest

# Add to .mcp.json
cat > .mcp.json << 'EOF'
{
  "mcpServers": {
    "quantconnect": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "QC_USER_ID",
        "-e", "QC_API_TOKEN",
        "quantconnect/mcp-server:latest"
      ]
    }
  }
}
EOF

# Set environment variables
export QC_USER_ID="your-user-id"
export QC_API_TOKEN="your-api-token"
```

### Alpaca MCP

```bash
# Install via npx (runs on demand)
npx -y @anthropic/alpaca-mcp --version

# Add to .mcp.json
cat >> .mcp.json << 'EOF'
{
  "mcpServers": {
    "alpaca": {
      "command": "npx",
      "args": ["-y", "@anthropic/alpaca-mcp"],
      "env": {
        "ALPACA_API_KEY": "${ALPACA_API_KEY}",
        "ALPACA_SECRET_KEY": "${ALPACA_SECRET_KEY}",
        "ALPACA_PAPER": "true"
      }
    }
  }
}
EOF

# Set environment variables
export ALPACA_API_KEY="your-api-key"
export ALPACA_SECRET_KEY="your-secret-key"
```

### Alpha Vantage MCP

```bash
# Install via npx
npx -y @anthropic/alphavantage-mcp --version

# Add to .mcp.json
# (append to existing mcpServers object)

# Set environment variable
export ALPHA_VANTAGE_API_KEY="your-api-key"
```

### Finviz MCP (Requires Elite)

```bash
# Install via uvx
pip install uv  # If not installed
uvx finviz-mcp --version

# Set environment variable
export FINVIZ_API_KEY="your-elite-api-key"
```

### Complete .mcp.json Example

```json
{
  "mcpServers": {
    "quantconnect": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "QC_USER_ID",
        "-e", "QC_API_TOKEN",
        "quantconnect/mcp-server:latest"
      ]
    },
    "alpaca": {
      "command": "npx",
      "args": ["-y", "@anthropic/alpaca-mcp"],
      "env": {
        "ALPACA_API_KEY": "${ALPACA_API_KEY}",
        "ALPACA_SECRET_KEY": "${ALPACA_SECRET_KEY}",
        "ALPACA_PAPER": "true"
      }
    },
    "alphavantage": {
      "command": "npx",
      "args": ["-y", "@anthropic/alphavantage-mcp"],
      "env": {
        "ALPHA_VANTAGE_API_KEY": "${ALPHA_VANTAGE_API_KEY}"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./"]
    }
  }
}
```

---

## Temporal.io Setup

### Step 1: Install Temporal CLI

```bash
# macOS
brew install temporal

# Linux
curl -sSf https://temporal.download/cli.sh | sh

# Windows (PowerShell)
# Download from https://temporal.io/downloads

# Verify installation
temporal --version
```

### Step 2: Start Development Server

```bash
# Start local Temporal server
temporal server start-dev

# Server runs at http://localhost:8233 (UI)
# Workflow endpoint: localhost:7233
```

### Step 3: Install Python SDK

```bash
pip install temporalio

# Add to requirements.txt
echo "temporalio>=1.0.0" >> requirements.txt
```

### Step 4: Create Workflow Directory

```bash
mkdir -p workflows

cat > workflows/__init__.py << 'PYTHON'
"""
Temporal workflows for autonomous trading research.
"""
from .research import TradingResearchWorkflow
from .backtest import BacktestWorkflow

__all__ = ["TradingResearchWorkflow", "BacktestWorkflow"]
PYTHON
```

### Step 5: Create Basic Workflow

```bash
cat > workflows/research.py << 'PYTHON'
"""Trading research workflow with Temporal durability."""
from datetime import timedelta
from temporalio import workflow, activity
from temporalio.common import RetryPolicy

@activity.defn
async def run_backtest(algorithm: str, params: dict) -> dict:
    """Execute backtest via LEAN CLI."""
    # Implementation here
    return {"sharpe": 1.5, "drawdown": 0.15}

@activity.defn
async def analyze_results(results: dict) -> dict:
    """Analyze backtest results."""
    return {"recommendation": "proceed" if results["sharpe"] > 1.0 else "review"}

@workflow.defn
class TradingResearchWorkflow:
    @workflow.run
    async def run(self, params: dict) -> dict:
        results = await workflow.execute_activity(
            run_backtest,
            args=[params["algorithm"], params],
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )

        analysis = await workflow.execute_activity(
            analyze_results,
            args=[results],
            start_to_close_timeout=timedelta(minutes=30)
        )

        return {"results": results, "analysis": analysis}
PYTHON
```

### Step 6: Create Worker

```bash
cat > workflows/worker.py << 'PYTHON'
"""Temporal worker for trading workflows."""
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from .research import TradingResearchWorkflow, run_backtest, analyze_results

async def main():
    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue="trading-research",
        workflows=[TradingResearchWorkflow],
        activities=[run_backtest, analyze_results],
    )

    print("Worker started, listening on task queue: trading-research")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
PYTHON
```

---

## Budget Tracking Setup

### Step 1: Install LiteLLM

```bash
pip install litellm

# Add to requirements.txt
echo "litellm>=1.0.0" >> requirements.txt
```

### Step 2: Configure Budget Tracking

```python
# In your startup script or watchdog
import litellm

# Set budget limits
litellm.max_budget = 50.0  # USD per day
litellm.budget_duration = "daily"

# Enable callbacks
litellm.success_callback = ["budget_tracking"]
litellm.failure_callback = ["budget_tracking"]
```

### Step 3: Create Budget Monitor

```bash
cat > scripts/budget_monitor.py << 'PYTHON'
#!/usr/bin/env python3
"""Monitor Claude API spending."""
import os
import json
from datetime import datetime
from pathlib import Path

BUDGET_FILE = Path("logs/budget.json")

def load_budget():
    if BUDGET_FILE.exists():
        return json.loads(BUDGET_FILE.read_text())
    return {"daily_limit": 50.0, "spent_today": 0.0, "last_reset": str(datetime.now().date())}

def save_budget(budget):
    BUDGET_FILE.parent.mkdir(exist_ok=True)
    BUDGET_FILE.write_text(json.dumps(budget, indent=2))

def check_budget():
    budget = load_budget()

    # Reset if new day
    today = str(datetime.now().date())
    if budget["last_reset"] != today:
        budget["spent_today"] = 0.0
        budget["last_reset"] = today
        save_budget(budget)

    remaining = budget["daily_limit"] - budget["spent_today"]
    print(f"Budget: ${budget['spent_today']:.2f} / ${budget['daily_limit']:.2f}")
    print(f"Remaining: ${remaining:.2f}")

    return remaining > 0

def record_spend(amount: float):
    budget = load_budget()
    budget["spent_today"] += amount
    save_budget(budget)

    if budget["spent_today"] >= budget["daily_limit"]:
        print("WARNING: Daily budget exhausted!")
        return False
    return True

if __name__ == "__main__":
    check_budget()
PYTHON

chmod +x scripts/budget_monitor.py
```

---

## Verification

### Run Complete Verification

```bash
# Create verification script
cat > scripts/verify_installation.sh << 'BASH'
#!/bin/bash
set -e

echo "=== Autonomous Claude Code Installation Verification ==="
echo

echo "1. Checking Claude Code..."
claude --version || { echo "FAIL: Claude Code not installed"; exit 1; }
echo "   OK"

echo "2. Checking Docker..."
docker --version || { echo "FAIL: Docker not installed"; exit 1; }
echo "   OK"

echo "3. Checking Docker Sandbox..."
docker sandbox --help > /dev/null 2>&1 || { echo "WARN: Docker Sandbox may not be available"; }
echo "   OK (or warning above)"

echo "4. Checking Python dependencies..."
python3 -c "import psutil; import litellm" || { echo "FAIL: Missing Python deps"; exit 1; }
echo "   OK"

echo "5. Checking project structure..."
[ -f ".claude/settings.json" ] || { echo "FAIL: .claude/settings.json missing"; exit 1; }
[ -f ".claude/hooks/protect_files.py" ] || { echo "FAIL: protect_files.py missing"; exit 1; }
[ -f "scripts/watchdog.py" ] || { echo "FAIL: watchdog.py missing"; exit 1; }
[ -f "scripts/run_overnight.sh" ] || { echo "FAIL: run_overnight.sh missing"; exit 1; }
echo "   OK"

echo "6. Checking MCP configuration..."
[ -f ".mcp.json" ] || { echo "WARN: .mcp.json missing - MCP servers not configured"; }
echo "   OK (or warning above)"

echo "7. Checking environment variables..."
[ -n "$QC_USER_ID" ] || echo "   WARN: QC_USER_ID not set"
[ -n "$QC_API_TOKEN" ] || echo "   WARN: QC_API_TOKEN not set"
echo "   Check complete"

echo
echo "=== Verification Complete ==="
echo "Run 'scripts/run_overnight.sh' to start an autonomous session"
BASH

chmod +x scripts/verify_installation.sh
./scripts/verify_installation.sh
```

### Quick Start After Installation

```bash
# 1. Start overnight session
./scripts/run_overnight.sh

# 2. Monitor progress
tail -f claude-progress.txt

# 3. Check watchdog logs
tail -f logs/watchdog.log

# 4. Check budget
python scripts/budget_monitor.py
```

---

## Troubleshooting

### Claude Code Issues

| Issue | Solution |
|-------|----------|
| `claude: command not found` | Re-run install script, check PATH |
| Authentication failed | Run `claude logout` then `claude login` |
| Rate limited | Wait for 5-hour window reset |

### Docker Issues

| Issue | Solution |
|-------|----------|
| `docker sandbox` not found | Update Docker Desktop to 4.50+ |
| Permission denied | Run `sudo usermod -aG docker $USER` |
| Container won't start | Check `docker logs` output |

### MCP Server Issues

| Issue | Solution |
|-------|----------|
| Server not connecting | Check `.mcp.json` syntax |
| Environment variable missing | Export variables or add to `.env` |
| npx fails | Clear npm cache: `npm cache clean --force` |

---

## Next Steps

After installation:

1. Review [TODO.md](TODO.md) for implementation checklist
2. Run verification script
3. Start first autonomous session
4. Monitor and adjust parameters

---

## Related Documentation

- [README.md](README.md) - Main guide
- [COMPARISON.md](COMPARISON.md) - Tool comparisons
- [TODO.md](TODO.md) - Implementation checklist
