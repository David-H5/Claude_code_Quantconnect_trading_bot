# UPGRADE-015: Continuous Implementation Guide

## Overview

**Created**: December 4, 2025
**Purpose**: Break down UPGRADE-015 into atomic micro-tasks for continuous autonomous execution
**Total Micro-Tasks**: 127
**Execution Mode**: Continuous until user stop - do NOT stop until explicitly told

---

## Execution Protocol

### Starting the Session

```bash
# Start continuous session - runs until manually stopped
./scripts/run_overnight.sh --continuous --with-recovery "UPGRADE-015: Autonomous AI Trading System Implementation"
```

### CRITICAL: Continuation Rules

1. **DO NOT STOP** until explicitly told by user
2. **MANDATORY**: Each task must be marked `[x]` before proceeding to next
3. **SEQUENTIAL**: Tasks must be completed in order listed below
4. **COMMIT**: Git commit after each major task completion
5. **DOCUMENT**: Update `claude-progress.txt` after each task
6. **NO SKIP**: Do not skip tasks, even if they seem trivial
7. **ON ERROR**: Document the error, attempt fix, continue to next task if blocked

### Progress Tracking

Update `claude-progress.txt` with:
```text
[Task X.Y] COMPLETE: [brief description]
[Task X.Y] BLOCKED: [reason] - continuing to next task
```

---

## CONTINUOUS TASK LIST

Execute all tasks below in sequence. Do not stop between phases.

### Phase 1: MCP Server Foundation

- [ ] **1.1** Create `mcp/` directory in project root
- [ ] **1.2** Create `mcp/__init__.py` with package exports
- [ ] **1.3** Create `mcp/base_server.py` - Base MCP server class with:
  - Standard initialization pattern
  - Error handling wrapper
  - Logging configuration
  - Health check endpoint
- [ ] **1.4** Create `mcp/schemas.py` - Pydantic models for:
  - QuoteRequest/QuoteResponse
  - OptionChainRequest/OptionChainResponse
  - OrderRequest/OrderResponse
  - PortfolioRequest/PortfolioResponse
- [ ] **1.5** Create `mcp/market_data_server.py` with tools:
  - `get_quote(symbol)` - Current stock quote
  - `get_option_chain(symbol, expiry_range)` - Option contracts
  - `get_greeks(contract)` - Delta, Gamma, Theta, Vega, Rho
  - `get_historical(symbol, period, resolution)` - OHLCV data
- [ ] **1.6** Add unit tests `tests/mcp/test_market_data_server.py`
- [ ] **1.7** Create `.mcp.json` configuration file with market-data server
- [ ] **1.8** Test market data MCP via Claude Code (manual verification)
- [ ] **1.9** Git commit: "feat(mcp): add market data server with quote and Greeks tools"

### Phase 2: Broker MCP Server

- [ ] **2.1** Create `mcp/broker_server.py` base structure
- [ ] **2.2** Implement `get_positions()` - List current positions
- [ ] **2.3** Implement `get_orders()` - List open orders
- [ ] **2.4** Implement `place_order(order_request)` - Submit new order
  - CRITICAL: Paper mode only (check `TRADING_MODE` env var)
  - Validate against risk limits before submission
- [ ] **2.5** Implement `cancel_order(order_id)` - Cancel pending order
- [ ] **2.6** Implement `get_fills()` - Recent fill history
- [ ] **2.7** Add order validation logic (max size, position limits)
- [ ] **2.8** Add unit tests `tests/mcp/test_broker_server.py`
- [ ] **2.9** Update `.mcp.json` with broker server configuration
- [ ] **2.10** Test broker MCP tools via Claude Code
- [ ] **2.11** Git commit: "feat(mcp): add broker server with paper trading support"

### Phase 3: Portfolio & Backtest MCP

- [ ] **3.1** Create `mcp/portfolio_server.py` with:
  - `get_portfolio_summary()` - Total value, cash, positions
  - `get_exposure()` - Sector/symbol exposure breakdown
  - `get_daily_pnl()` - Today's P&L
  - `get_risk_metrics()` - VaR, max drawdown, Sharpe
- [ ] **3.2** Create `mcp/backtest_server.py` with:
  - `run_backtest(algorithm, start_date, end_date)` - Execute LEAN backtest
  - `get_backtest_results(backtest_id)` - Parse results
  - `compare_backtests(ids)` - Compare multiple runs
- [ ] **3.3** Create backtest result parser (extract Sharpe, drawdown, returns)
- [ ] **3.4** Add tests for portfolio server
- [ ] **3.5** Add tests for backtest server
- [ ] **3.6** Update `.mcp.json` with all 4 servers
- [ ] **3.7** Create `mcp/README.md` documentation
- [ ] **3.8** End-to-end test all MCP servers together
- [ ] **3.9** Git commit: "feat(mcp): complete MCP ecosystem with portfolio and backtest servers"

### Phase 4: Hook System Implementation

- [ ] **4.1** Create `.claude/hooks/` directory if not exists
- [ ] **4.2** Create `risk_validator.py` PreToolUse hook:
  - Parse incoming order from stdin
  - Check against CircuitBreaker state
  - Validate position limits (max 25% per position)
  - Validate daily loss limits (max 3%)
  - Exit 0 to allow, Exit 2 to block with message
- [ ] **4.3** Create `log_trade.py` PostToolUse hook:
  - Log all broker tool calls to JSONL file
  - Include timestamp, tool name, inputs, outputs
  - Rotate logs daily
- [ ] **4.4** Create `load_context.py` SessionStart hook (adapt from template):
  - Load portfolio state from `data/portfolio_state.json`
  - Check market status (open/closed)
  - Load recent trading activity
  - Display session banner
- [ ] **4.5** Create `parse_backtest.py` PostToolUse hook:
  - Match `Bash(lean backtest *)` commands
  - Extract key metrics (Sharpe, drawdown, return)
  - Output structured JSON for Claude
- [ ] **4.6** Create `algo_change_guard.py` PreToolUse hook:
  - Match `Write(algorithms/**)` or `Edit(algorithms/**)`
  - Require confirmation for algorithm file changes
  - Log all algorithm modifications
- [ ] **4.7** Update `.claude/settings.json` with all hook configurations
- [ ] **4.8** Add hook unit tests `tests/hooks/test_risk_validator.py`
- [ ] **4.9** Add hook unit tests `tests/hooks/test_log_trade.py`
- [ ] **4.10** Test hook blocking behavior manually
- [ ] **4.11** Git commit: "feat(hooks): implement trading safety hooks with risk validation"

### Phase 5: Agent Personas

- [ ] **5.1** Create `.claude/agents/` directory
- [ ] **5.2** Create `senior-engineer.md`:
  - Focus: Production-quality code, best practices
  - Tools: Read, Write, Edit, Bash, Grep, Glob
  - Constraints: Must include tests, type hints, docstrings
- [ ] **5.3** Create `risk-reviewer.md`:
  - Focus: Risk and compliance review
  - Tools: Read, Grep, Glob (read-only)
  - Constraints: Flag security issues, validate limits
- [ ] **5.4** Create `strategy-dev.md`:
  - Focus: Algorithm development and optimization
  - Tools: Full access
  - Constraints: Must backtest before suggesting changes
- [ ] **5.5** Create `code-reviewer.md`:
  - Focus: PR review and code quality
  - Tools: Read, Grep, Glob
  - Constraints: Check for bugs, security, performance
- [ ] **5.6** Create `qa-engineer.md`:
  - Focus: Testing and quality assurance
  - Tools: Full access
  - Constraints: Write comprehensive tests, verify coverage
- [ ] **5.7** Create `researcher.md`:
  - Focus: Market research and analysis
  - Tools: Read, Grep, WebSearch, WebFetch
  - Constraints: Document all findings with timestamps
- [ ] **5.8** Create `backtest-analyst.md`:
  - Focus: Backtest analysis and optimization
  - Tools: Read, Grep, Glob, Bash(lean:*)
  - Constraints: Statistical rigor, avoid overfitting
- [ ] **5.9** Add agent usage guide to CLAUDE.md
- [ ] **5.10** Test each persona via Task tool invocation
- [ ] **5.11** Git commit: "feat(agents): add 7 specialized agent personas for subagent workflows"

### Phase 6: Observability Setup

- [ ] **6.1** Add `agentops` to requirements.txt
- [ ] **6.2** Create `observability/` directory
- [ ] **6.3** Create `observability/__init__.py` with exports
- [ ] **6.4** Create `observability/agentops_client.py`:
  - Initialize AgentOps with API key from env
  - Create session wrapper
  - Add cost tracking methods
  - Add failure detection methods
- [ ] **6.5** Create `observability/token_tracker.py`:
  - Track input/output tokens per call
  - Calculate cumulative costs
  - Alert when thresholds exceeded
  - Daily/weekly/monthly rollups
- [ ] **6.6** Create `observability/decision_tracer.py`:
  - Trace decision paths through agent reasoning
  - Log confidence scores at each step
  - Identify decision bottlenecks
- [ ] **6.7** Create `observability/metrics.py`:
  - Define Prometheus-compatible metrics
  - Trading metrics (fill rate, slippage, P&L)
  - Agent metrics (latency, token usage, errors)
- [ ] **6.8** Add tests `tests/observability/test_token_tracker.py`
- [ ] **6.9** Add tests `tests/observability/test_metrics.py`
- [ ] **6.10** Update `.env.example` with AgentOps API key placeholder
- [ ] **6.11** Create `observability/README.md` documentation
- [ ] **6.12** Git commit: "feat(observability): add AgentOps and token tracking integration"

### Phase 7: Redis Infrastructure

- [ ] **7.1** Add `redis` to requirements.txt
- [ ] **7.2** Create `infrastructure/` directory
- [ ] **7.3** Create `infrastructure/redis_client.py`:
  - Connection pool management
  - Health check method
  - Reconnection with backoff
- [ ] **7.4** Create `infrastructure/market_stream.py`:
  - Ingest market data to Redis Streams
  - Consumer group setup
  - Message acknowledgment
- [ ] **7.5** Create `infrastructure/timeseries.py`:
  - OHLCV storage with Redis TimeSeries
  - Aggregation queries (1m, 5m, 1h, 1d)
  - Retention policy (30 days)
- [ ] **7.6** Create `infrastructure/pubsub.py`:
  - Price change notifications
  - Alert broadcasting
  - Subscriber management
- [ ] **7.7** Add `docker-compose.redis.yml` for local Redis
- [ ] **7.8** Add tests `tests/infrastructure/test_redis_client.py`
- [ ] **7.9** Add tests `tests/infrastructure/test_market_stream.py`
- [ ] **7.10** Create `infrastructure/README.md`
- [ ] **7.11** Git commit: "feat(infrastructure): add Redis Streams and TimeSeries for real-time data"

### Phase 8: Options Analytics Engine

- [ ] **8.1** Create `analytics/` directory
- [ ] **8.2** Create `analytics/__init__.py` with exports
- [ ] **8.3** Create `analytics/iv_surface.py`:
  - Build strike x expiry x IV matrix
  - Interpolation using scipy Rbf
  - Extrapolation guards
  - Volatility smile visualization data
- [ ] **8.4** Create `analytics/term_structure.py`:
  - Extract term structure from surface
  - Constant maturity interpolation
  - Term structure slope indicator
- [ ] **8.5** Create `analytics/volatility_skew.py`:
  - Calculate skew at each expiry
  - Detect skew anomalies
  - Skew-based trading signals
- [ ] **8.6** Create `analytics/greeks_calculator.py`:
  - Black-Scholes implementation
  - IV-based Greeks (matching QuantConnect)
  - Greeks sensitivity analysis
- [ ] **8.7** Create `analytics/pricing_models.py`:
  - Black-Scholes-Merton
  - Bjerksund-Stensland (American)
  - Model selection logic
- [ ] **8.8** Add tests `tests/analytics/test_iv_surface.py`
- [ ] **8.9** Add tests `tests/analytics/test_greeks_calculator.py`
- [ ] **8.10** Create `analytics/README.md`
- [ ] **8.11** Git commit: "feat(analytics): add IV surface construction and advanced Greeks"

### Phase 9: Backtesting Robustness

- [ ] **9.1** Create `backtesting/` directory
- [ ] **9.2** Create `backtesting/__init__.py` with exports
- [ ] **9.3** Create `backtesting/walk_forward.py`:
  - Rolling train/test window
  - Parameter optimization per window
  - Out-of-sample validation tracking
  - Aggregated performance metrics
- [ ] **9.4** Create `backtesting/monte_carlo.py`:
  - Bootstrap trade sequences
  - Calculate drawdown distribution
  - Risk metrics (P5, P50, P95 drawdown)
  - Confidence intervals
- [ ] **9.5** Create `backtesting/parameter_sensitivity.py`:
  - Grid search implementation
  - Variance analysis per parameter
  - Stability scoring
- [ ] **9.6** Create `backtesting/regime_detector.py`:
  - Bull/bear/sideways classification
  - Regime-specific performance
  - Regime change detection
- [ ] **9.7** Create `backtesting/overfitting_guard.py`:
  - Variance-based overfitting detection
  - Complexity penalty (parameter count)
  - Recommendation engine
- [ ] **9.8** Add tests `tests/backtesting/test_walk_forward.py`
- [ ] **9.9** Add tests `tests/backtesting/test_monte_carlo.py`
- [ ] **9.10** Create `backtesting/README.md`
- [ ] **9.11** Git commit: "feat(backtesting): add walk-forward optimization and Monte Carlo simulation"

### Phase 10: Multi-Agent Architecture

- [ ] **10.1** Create `agents/` directory (separate from .claude/agents/)
- [ ] **10.2** Create `agents/__init__.py` with exports
- [ ] **10.3** Create `agents/base_agent.py`:
  - Base class for all trading agents
  - Standard interface (analyze, decide, execute)
  - State management
  - Communication protocol
- [ ] **10.4** Create `agents/orchestrator.py`:
  - Queen agent for coordination
  - Task decomposition logic
  - Worker spawning
  - Result synthesis
- [ ] **10.5** Create `agents/market_agent.py`:
  - Real-time market data processing
  - Opportunity detection
  - Signal generation
- [ ] **10.6** Create `agents/strategy_agent.py`:
  - Strategy execution logic
  - Entry/exit decisions
  - Position sizing
- [ ] **10.7** Create `agents/risk_agent.py`:
  - Real-time risk monitoring
  - Limit enforcement
  - Kill switch integration
- [ ] **10.8** Create `agents/communication.py`:
  - Inter-agent message protocol
  - Event broadcasting
  - State synchronization
- [ ] **10.9** Add tests `tests/agents/test_orchestrator.py`
- [ ] **10.10** Add tests `tests/agents/test_communication.py`
- [ ] **10.11** Create `agents/README.md`
- [ ] **10.12** Git commit: "feat(agents): implement hierarchical multi-agent architecture"

### Phase 11: Compliance & Audit Logging

- [ ] **11.1** Create `compliance/` directory
- [ ] **11.2** Create `compliance/__init__.py` with exports
- [ ] **11.3** Create `compliance/audit_logger.py`:
  - Immutable append-only log structure
  - Required fields (timestamp, actor, action, resource, outcome)
  - Log rotation with retention (7 years)
  - Digital signature for integrity
- [ ] **11.4** Create `compliance/anti_manipulation.py`:
  - Wash sale detection
  - Spoofing pattern detection
  - Self-trade prevention
  - Alert generation
- [ ] **11.5** Create `compliance/reporting.py`:
  - Daily compliance report generator
  - Trade summary reports
  - Exception reports
  - Export to CSV/PDF
- [ ] **11.6** Create `compliance/retention_policy.py`:
  - Log archival automation
  - Compression for old logs
  - Retention period enforcement
  - Archive retrieval
- [ ] **11.7** Create `compliance/finra_checklist.py`:
  - FINRA compliance validation
  - Checklist generator
  - Deficiency reporting
- [ ] **11.8** Add tests `tests/compliance/test_audit_logger.py`
- [ ] **11.9** Add tests `tests/compliance/test_anti_manipulation.py`
- [ ] **11.10** Create `compliance/README.md` with FINRA reference
- [ ] **11.11** Git commit: "feat(compliance): add audit logging and anti-manipulation detection"

### Phase 12: Integration & Final Validation

- [ ] **12.1** Create `integration_tests/` directory
- [ ] **12.2** Create `integration_tests/test_mcp_full_flow.py`:
  - Test all 4 MCP servers together
  - Verify data consistency
- [ ] **12.3** Create `integration_tests/test_agent_coordination.py`:
  - Test multi-agent workflow
  - Verify orchestrator coordination
- [ ] **12.4** Create `integration_tests/test_safety_hooks.py`:
  - Test risk validation prevents bad orders
  - Test audit logging captures all actions
- [ ] **12.5** Create `integration_tests/test_analytics_pipeline.py`:
  - Test IV surface → Greeks → Strategy flow
- [ ] **12.6** Update CLAUDE.md with:
  - All new MCP tools
  - New agent personas
  - New slash commands
  - Updated architecture diagram
- [ ] **12.7** Update `docs/PROJECT_STATUS.md` with UPGRADE-015 completion
- [ ] **12.8** Create `docs/research/UPGRADE-015-COMPLETION-REPORT.md`:
  - Summary of all implemented features
  - Test coverage report
  - Performance metrics
  - Known limitations
- [ ] **12.9** Run full test suite: `pytest tests/ -v --cov=. --cov-fail-under=70`
- [ ] **12.10** Run linting: `ruff check . && mypy --config-file mypy.ini`
- [ ] **12.11** Create final git commit: "feat(upgrade-015): complete autonomous AI trading system upgrade"
- [ ] **12.12** Create git tag: `v2.0.0-upgrade-015`

---

## Task Dependencies

```
Phase 1 (MCP Foundation) ──► Phase 2 (Broker) ──► Phase 3 (Portfolio/Backtest)
                                      │
                                      ▼
Phase 4 (Hooks) ◄──────────────────┘
        │
        ▼
Phase 5 (Personas) ──► Phase 6 (Observability) ──► Phase 7 (Redis)
                                                        │
                                                        ▼
Phase 8 (Analytics) ──► Phase 9 (Backtesting) ──► Phase 10 (Multi-Agent)
                                                        │
                                                        ▼
                              Phase 11 (Compliance) ──► Phase 12 (Integration)
```

---

## Summary

| Metric | Value |
|--------|-------|
| Total Phases | 12 |
| Total Micro-Tasks | 127 |
| New Directories | ~10 |
| New Files | ~45 |
| New Test Files | ~20 |
| Documentation Files | ~10 |

---

## Change Log

| Date | Change | Impact |
|------|--------|--------|
| 2025-12-04 | Created continuous implementation guide | 127 micro-tasks, single continuous session |
