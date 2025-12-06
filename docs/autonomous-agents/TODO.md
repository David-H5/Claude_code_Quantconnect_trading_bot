# Autonomous Claude Code Implementation Checklist

**Version**: 1.1.0 | **Last Updated**: November 28, 2025

Use this checklist to track implementation progress. Check off items as you complete them.

---

## ⭐ Priority: November 2025 Releases

These new features should be implemented first - they significantly simplify autonomous development:

### Claude Code Native Sandbox (TOP PRIORITY)
- [ ] Verify Claude Code version supports `--sandbox` flag
- [ ] Test `/sandbox` command in interactive mode
- [ ] Test `claude --sandbox -p "task"` for headless mode
- [ ] Configure `--sandbox-allow-dirs` for project directory
- [ ] Configure `--sandbox-allow-network` for QuantConnect, GitHub
- [ ] Verify 84% permission reduction works as expected
- [ ] Update overnight script to use native sandbox instead of Docker

### Claude Code on Web
- [ ] Access Claude Code at claude.ai
- [ ] Connect GitHub repository
- [ ] Test remote task initiation from mobile/tablet
- [ ] Verify isolated sandbox environment

### Kiro GA (Optional - Spec-Driven Development)
- [ ] Install Kiro CLI (`npm install -g @kiro/cli` or download IDE)
- [ ] Test free tier (50 credits/month)
- [ ] Create requirements.md for trading strategy
- [ ] Create design.md for technical architecture
- [ ] Generate tasks.md from specs
- [ ] Test property-based testing for trading logic

### GitHub Copilot Subagents (Optional - CI/CD)
- [ ] Ensure GitHub Copilot Pro/Pro+/Business/Enterprise subscription
- [ ] Test assigning GitHub issue to Copilot
- [ ] Verify background PR generation
- [ ] Test isolated subagents in JetBrains/VSCode

---

## Phase 1: Foundation (Required)

### Claude Code Setup
- [ ] Install Claude Code CLI
- [ ] Authenticate with `claude login`
- [ ] Verify with `claude whoami`
- [ ] Choose plan: Max 5x (better Opus value) or Max 20x (more Sonnet hours)

### Project Configuration
- [ ] Create `.claude/` directory
- [ ] Create `.claude/settings.json` with permissions
- [ ] Create `.claude/hooks/` directory
- [ ] Create `.claude/hooks/protect_files.py`
- [ ] Test file protection hook works (try reading .env)
- [ ] Verify `deny` rules workaround is active

### Environment Variables
- [ ] Set `QC_USER_ID` for QuantConnect
- [ ] Set `QC_API_TOKEN` for QuantConnect
- [ ] Set `ANTHROPIC_API_KEY` (if using API directly)
- [ ] Create `.env` file (add to .gitignore!)
- [ ] Test environment variables are loaded

---

## Phase 2: Safety Infrastructure (Required)

### Watchdog Process
- [ ] Install `psutil` package
- [ ] Install `litellm` package
- [ ] Create `scripts/watchdog.py`
- [ ] Configure watchdog parameters in `config/watchdog.json`
- [ ] Test watchdog starts and monitors correctly
- [ ] Test watchdog terminates on timeout
- [ ] Test watchdog terminates on idle

### Budget Tracking
- [ ] Configure LiteLLM budget limits
- [ ] Create `scripts/budget_monitor.py`
- [ ] Create `logs/budget.json` tracking file
- [ ] Test budget tracking records spending
- [ ] Test budget alerts on threshold

### Circuit Breaker
- [ ] Verify `models/circuit_breaker.py` exists
- [ ] Configure circuit breaker parameters
- [ ] Test circuit breaker trips on conditions
- [ ] Test circuit breaker requires human reset
- [ ] Integrate circuit breaker with autonomous loop

### Progress Tracking
- [ ] Create `claude-progress.txt` template
- [ ] Configure checkpoint commits
- [ ] Test progress file updates correctly
- [ ] Test session can resume from progress file

---

## Phase 3: Sandbox Environment (Recommended)

### Docker Setup
- [ ] Install Docker Desktop 4.50+
- [ ] Verify Docker version with `docker --version`
- [ ] Test `docker sandbox --help` works
- [ ] Create Docker sandbox configuration
- [ ] Test `docker sandbox run claude` works
- [ ] Test workspace mount works

### Resource Limits
- [ ] Configure memory limits (4GB recommended)
- [ ] Configure CPU limits (2 cores recommended)
- [ ] Configure network policy
- [ ] Test resource limits are enforced

### Alternative: Microsandbox (Self-Hosted)
- [ ] Install Microsandbox (if not using Docker Sandbox)
- [ ] Configure microVM settings
- [ ] Test network isolation
- [ ] Test filesystem isolation

---

## Phase 4: MCP Servers (Recommended)

### QuantConnect MCP (Primary)
- [ ] Pull `quantconnect/mcp-server` Docker image
- [ ] Configure in `.mcp.json`
- [ ] Set `QC_USER_ID` environment variable
- [ ] Set `QC_API_TOKEN` environment variable
- [ ] Test connection with simple query
- [ ] Test backtest execution via MCP

### Alpaca MCP (Paper Trading)
- [ ] Create Alpaca paper trading account
- [ ] Install `@anthropic/alpaca-mcp`
- [ ] Configure in `.mcp.json`
- [ ] Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`
- [ ] Enable paper trading mode
- [ ] Test quote retrieval
- [ ] Test paper order placement

### Alpha Vantage MCP (Market Data)
- [ ] Get Alpha Vantage API key (free tier)
- [ ] Install `@anthropic/alphavantage-mcp`
- [ ] Configure in `.mcp.json`
- [ ] Test stock data retrieval
- [ ] Test technical indicator retrieval

### Finviz MCP (Optional - Requires Elite)
- [ ] Subscribe to Finviz Elite ($40/month)
- [ ] Get Finviz API key
- [ ] Install `finviz-mcp` via uvx
- [ ] Configure in `.mcp.json`
- [ ] Test stock screener
- [ ] Test SEC filings retrieval
- [ ] Test insider trading data

---

## Phase 5: Overnight Session Scripts (Required)

### Startup Script
- [ ] Create `scripts/run_overnight.sh`
- [ ] Configure pre-flight checks
- [ ] Configure watchdog auto-start
- [ ] Configure logging directory
- [ ] Configure checkpoint interval
- [ ] Make script executable (`chmod +x`)
- [ ] Test script runs correctly

### Session Management
- [ ] Configure context limit (60% rule)
- [ ] Configure auto-compact
- [ ] Create session handoff template
- [ ] Test session continuation works
- [ ] Test context summarization

### Monitoring
- [ ] Configure log rotation
- [ ] Create `logs/` directory
- [ ] Set up progress file monitoring
- [ ] Set up cost monitoring
- [ ] Configure alerts (email/Slack optional)

---

## Phase 6: Durable Execution (Optional - Advanced)

### Temporal.io Setup
- [ ] Install Temporal CLI
- [ ] Start Temporal development server
- [ ] Install `temporalio` Python package
- [ ] Create `workflows/` directory
- [ ] Create basic research workflow
- [ ] Create worker script
- [ ] Test workflow execution
- [ ] Test workflow retry on failure
- [ ] Test workflow state persistence

### Workflow Integration
- [ ] Create backtest workflow
- [ ] Create strategy analysis workflow
- [ ] Create optimization workflow
- [ ] Integrate with Claude Code session
- [ ] Test end-to-end workflow

---

## Phase 7: Multi-Agent Architecture (Optional - Advanced)

### Agent Framework Selection
- [ ] Evaluate Claude Agent SDK
- [ ] Evaluate LangGraph
- [ ] Evaluate CrewAI
- [ ] Select primary framework
- [ ] Install selected framework

### Agent Design
- [ ] Define agent roles (Scanner, Analyst, Executor)
- [ ] Design agent communication protocol
- [ ] Design supervisor pattern
- [ ] Create agent configuration files

### Implementation
- [ ] Create Scanner agent
- [ ] Create Analysis agent
- [ ] Create Execution agent
- [ ] Create Supervisor/Orchestrator
- [ ] Test agent collaboration
- [ ] Test error handling between agents

---

## Phase 8: CI/CD Integration (Optional)

### GitHub Actions
- [ ] Configure Claude Code review workflow
- [ ] Configure autonomous testing workflow
- [ ] Configure backtest workflow
- [ ] Configure paper trading deployment
- [ ] Set up required secrets

### Branch Protection
- [ ] Configure main branch protection
- [ ] Configure develop branch protection
- [ ] Require passing tests for merge
- [ ] Require backtest approval for main

---

## Verification Checklist

### Before First Overnight Session
- [ ] All Phase 1 items complete
- [ ] All Phase 2 items complete
- [ ] Watchdog tested and working
- [ ] Budget limits configured
- [ ] Circuit breaker configured
- [ ] Progress file template ready
- [ ] Logs directory exists
- [ ] Test session runs for 30 minutes without issues

### Before Production Use
- [ ] At least 3 successful overnight sessions
- [ ] Watchdog has correctly terminated at least once
- [ ] Budget tracking is accurate
- [ ] Progress files are readable and resumable
- [ ] No unexpected costs incurred
- [ ] All safety mechanisms tested

---

## Quick Reference

### Commands

```bash
# NEW (November 2025): Start with native sandbox
claude --sandbox -p "Implement feature X" --output-format stream-json

# Enable sandbox in interactive mode
/sandbox

# Sandbox with network access
claude --sandbox --sandbox-allow-network "api.quantconnect.com,github.com"

# Start overnight session (legacy Docker)
./scripts/run_overnight.sh

# Check watchdog status
ps aux | grep watchdog

# Monitor progress
tail -f claude-progress.txt

# Check budget
python scripts/budget_monitor.py

# Verify installation
./scripts/verify_installation.sh

# Run in Docker sandbox (alternative)
docker sandbox run -w $(pwd) claude
```

### Files to Create

| File | Purpose | Status |
|------|---------|--------|
| `.claude/settings.json` | Claude Code permissions | [x] |
| `.claude/hooks/protect_files.py` | File protection hook | [x] |
| `.mcp.json` | MCP server configuration | [ ] |
| `scripts/watchdog.py` | External watchdog | [x] |
| `scripts/run_overnight.sh` | Startup script | [x] |
| `scripts/budget_monitor.py` | Cost tracking | [ ] |
| `config/watchdog.json` | Watchdog config | [x] |
| `claude-progress.txt` | Session progress | [ ] |
| `AGENTS.md` | Universal agent instructions | [ ] |
| `docs/autonomous-agents/README.md` | Main autonomous guide | [x] |
| `docs/autonomous-agents/COMPARISON.md` | Tool comparisons | [x] |
| `docs/autonomous-agents/INSTALLATION.md` | Setup instructions | [x] |
| `docs/autonomous-agents/TODO.md` | This checklist | [x] |

---

## Phase 9: Loop Workflows (Optional - Advanced)

### continuous-claude Setup
- [ ] Clone continuous-claude repository
- [ ] Configure GOAL.md with project objectives
- [ ] Set iteration limits (default 50)
- [ ] Configure CI checks integration
- [ ] Test single iteration
- [ ] Test overnight run (limited iterations)

### claude-flow Multi-Agent (Enterprise)
- [ ] Install claude-flow alpha
- [ ] Initialize project (`npx claude-flow@alpha init`)
- [ ] Configure agent swarm architecture
- [ ] Test single agent task
- [ ] Test multi-agent coordination

### Alternative Workflow Options
- [ ] Evaluate ralph-claude-code (exit detection)
- [ ] Evaluate claude-code-workflow (spec-driven)
- [ ] Select best fit for project needs

---

## Phase 10: Standards Compliance (Recommended)

### AGENTS.md Standard
- [ ] Create `AGENTS.md` at project root
- [ ] Add project overview section
- [ ] Add build & test instructions
- [ ] Add code style guidelines
- [ ] Add security requirements
- [ ] Add architecture description
- [ ] Test with multiple agents (Cursor, Copilot)

### OpenTelemetry Observability
- [ ] Install opentelemetry SDK
- [ ] Configure GenAI semantic conventions
- [ ] Add agent tracing spans
- [ ] Set up log aggregation
- [ ] Create monitoring dashboard

### Environment Variables

| Variable | Required | Source |
|----------|----------|--------|
| `QC_USER_ID` | Yes | QuantConnect account |
| `QC_API_TOKEN` | Yes | QuantConnect API settings |
| `ALPACA_API_KEY` | Optional | Alpaca dashboard |
| `ALPACA_SECRET_KEY` | Optional | Alpaca dashboard |
| `ALPHA_VANTAGE_API_KEY` | Optional | Alpha Vantage signup |
| `FINVIZ_API_KEY` | Optional | Finviz Elite subscription |

---

## Progress Log

Use this section to track your progress:

| Date | Phase | Items Completed | Notes |
|------|-------|-----------------|-------|
| | | | |
| | | | |
| | | | |

---

## Related Documentation

- [README.md](README.md) - Main guide
- [COMPARISON.md](COMPARISON.md) - Tool comparisons
- [INSTALLATION.md](INSTALLATION.md) - Setup instructions
