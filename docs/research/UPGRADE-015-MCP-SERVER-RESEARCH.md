# UPGRADE-015: MCP Server Implementation Research

## Research Overview

**Date**: December 4, 2025
**Scope**: Model Context Protocol (MCP) server implementation for trading system
**Focus**: Python SDK, Claude Code integration, trading MCP patterns
**Result**: Comprehensive implementation guide for MCP trading servers

---

## Research Objectives

1. Understand MCP Python SDK architecture and best practices
2. Learn Claude Code MCP integration patterns
3. Research existing trading MCP server implementations
4. Document implementation approach for UPGRADE-015

---

## Research Phases

### Phase 1: MCP Python SDK Research

**Search Date**: December 4, 2025 at 17:10 EST
**Search Query**: "Model Context Protocol MCP Python SDK implementation 2025"

**Key Sources**:

1. [Official Python SDK GitHub (Published: Nov 2024, Updated: Dec 2025)](https://github.com/modelcontextprotocol/python-sdk) - 20,400+ stars
2. [MCP PyPI Package (Published: Dec 2025)](https://pypi.org/project/mcp/) - Version 1.23.1
3. [MCP Wikipedia (Updated: 2025)](https://en.wikipedia.org/wiki/Model_Context_Protocol)
4. [MCP Complete Tutorial - Medium (Published: 2025)](https://medium.com/@nimritakoul01/the-model-context-protocol-mcp-a-complete-tutorial-a3abe8a7f4ef)
5. [OpenAI Agents SDK MCP (Published: 2025)](https://openai.github.io/openai-agents-python/mcp/)

**Key Discoveries**:

1. **Current Version**: MCP Python SDK 1.23.1 (spec 2025-11-25)
2. **Transport Evolution**:
   - 2024-11-05: HTTP with SSE
   - 2025-03-26: Streamable HTTP (bidirectional)
   - 2025-06-18: Tightened security, removed batching
3. **Core Capabilities**:
   - Resources: File-like data exposure
   - Tools: Functions callable by LLM
   - Prompts: Pre-written templates
4. **Authorization**: OAuth 2.1 resource server (RFC 9728)
5. **Adoption**: OpenAI adopted MCP in March 2025

**Applied**: Will use mcp-1.23.1 for implementation

---

### Phase 2: Claude Code MCP Integration

**Search Date**: December 4, 2025 at 17:10 EST
**Search Query**: "Claude Code MCP server tutorial tools implementation"

**Key Sources**:

1. [Official Claude Code MCP Docs (Published: 2025)](https://docs.claude.com/en/docs/claude-code/mcp)
2. [Scott Spence MCP Config Guide (Published: 2025)](https://scottspence.com/posts/configuring-mcp-tools-in-claude-code)
3. [MCP Server Quickstart (Published: 2025)](https://modelcontextprotocol.io/quickstart/server)
4. [Docker MCP Toolkit (Published: 2025)](https://www.docker.com/blog/add-mcp-servers-to-claude-code-with-mcp-toolkit/)
5. [DigitalOcean MCP Tutorial (Published: 2025)](https://www.digitalocean.com/community/tutorials/claude-code-mcp-server)

**Key Discoveries**:

1. **Configuration Methods**:
   - CLI: `claude mcp add [name] --scope user`
   - Direct config file editing (preferred for complex setups)
2. **Token Limits**:
   - Warning at 10,000 tokens
   - Max 25,000 tokens (adjustable via MAX_MCP_OUTPUT_TOKENS)
3. **Timeout**: MCP_TIMEOUT env var for startup timeout
4. **Core JSON-RPC Methods**:
   - `initialize`: Connection + capabilities
   - `tools/list`: Available tools and schemas
   - `tools/call`: Execute tools with parameters
5. **Docker Toolkit**: 200+ pre-built containerized MCP servers

**Applied**: Will configure .mcp.json for project MCP servers

---

### Phase 3: Trading MCP Server Patterns

**Search Date**: December 4, 2025 at 17:10 EST
**Search Query**: "MCP trading system market data server Python example"

**Key Sources**:

1. [Alpaca MCP Server - Official (Published: 2025)](https://github.com/alpacahq/alpaca-mcp-server)
2. [Alpha Vantage MCP Tutorial (Published: 2025)](https://medium.com/@syed_hasan/step-by-step-guide-building-an-mcp-server-using-python-sdk-alphavantage-claude-ai-7a2bfb0c3096)
3. [Financial Datasets MCP Server (Published: 2025)](https://github.com/financial-datasets/mcp-server)
4. [FastAPI-MCP Stock Analysis (Published: 2025)](https://dev.to/mrzaizai2k/building-an-mcp-server-with-fastapi-mcp-for-stock-analysis-a-step-by-step-guide-de6)
5. [MetaTrader 5 MCP Server (Published: 2025)](https://www.mcp.pizza/mcp-server/7QfT/mcp-metatrader5-server)

**Key Discoveries**:

1. **Existing Trading MCP Servers**:
   - Alpaca: Stocks, options, crypto, portfolio, real-time data
   - Alpha Vantage: Stock data, moving averages, RSI
   - Financial Datasets: Income statements, balance sheets, prices
   - MetaTrader 5: Real-time data, trade execution, history
2. **Requirements**: Python 3.10+, mcp SDK 1.2.0+
3. **Best Practices**:
   - Use FastMCP for simpler server creation
   - Separate market data from trading execution
   - Include paper trading mode for safety
4. **Architecture Pattern**:
   - Resources for exposing data
   - Tools for executing actions
   - Prompts for templating interactions

**Applied**: Will follow Alpaca pattern with separate market data and broker servers

---

## Critical Discoveries

| Discovery | Impact | Source |
|-----------|--------|--------|
| MCP SDK 1.23.1 is current | Use latest SDK | PyPI Dec 2025 |
| OAuth 2.1 built into SDK | Use for auth | GitHub Python SDK |
| Paper trading mode critical | Safety first | Alpaca MCP Server |
| FastMCP simplifies development | Use for base server | FastAPI-MCP Tutorial |
| 25K token limit | Design for efficiency | Claude Code Docs |

---

## Implementation Plan

### MCP Server Architecture for UPGRADE-015

```
mcp/
├── __init__.py           # Package exports
├── base_server.py        # Base MCP server class (FastMCP)
├── schemas.py            # Pydantic models for requests/responses
├── market_data_server.py # Quote, option chain, Greeks, historical
├── broker_server.py      # Positions, orders, place_order, cancel
├── portfolio_server.py   # Summary, exposure, P&L, risk metrics
├── backtest_server.py    # Run backtest, get results, compare
└── README.md             # Documentation
```

### Key Implementation Patterns

1. **Use FastMCP** for simplified server creation:
   ```python
   from mcp.server.fastmcp import FastMCP
   mcp = FastMCP("Market Data Server")

   @mcp.tool()
   def get_quote(symbol: str) -> dict:
       """Get current stock quote."""
       pass
   ```

2. **Separate servers** by responsibility:
   - market-data: Read-only market data access
   - broker: Trading operations (paper mode enforced)
   - portfolio: Portfolio analytics
   - backtest: Backtesting operations

3. **Paper Trading Safety**:
   ```python
   @mcp.tool()
   def place_order(order: OrderRequest) -> OrderResponse:
       if os.environ.get("TRADING_MODE") != "paper":
           raise ToolError("Live trading disabled")
       # Execute paper order
   ```

4. **Token Efficiency**:
   - Paginate large responses
   - Return summaries, not raw data
   - Use streaming for real-time data

---

## Research Deliverables

| Deliverable | Status |
|-------------|--------|
| MCP SDK version identified | Complete (1.23.1) |
| Architecture pattern documented | Complete |
| Safety patterns identified | Complete (paper mode) |
| Implementation plan created | Complete |
| Research document persisted | Complete |

---

## Change Log

| Date | Change | Impact |
|------|--------|--------|
| 2025-12-04 | Initial research complete | Ready for Phase 1 implementation |

---

### Research Entry - December 04, 2025 at 10:47 PM

**Topic**: RIC v5.0 P4 REFLECT Introspection Gaming Fix
**Result**: Implemented RIC v5.1 Quality Gates

**Search Queries**:
- "LLM reflection introspection quality enforcement techniques 2025"
- "AI agent self-improvement loop iterative refinement enforcement 2025"
- "preventing AI agents gaming evaluation metrics Goodhart's Law solutions"
- "Reflexion framework LLM agent implementation feedback loop architecture"
- "SELF-REFINE paper iterative output critique mechanism how it works"

**Key Sources**:

1. [SELF-REFINE Paper (Published: March 2023)](https://arxiv.org/abs/2303.17651)
   - 3-step process: Initial → Feedback → Refine
   - Feedback must be ACTIONABLE with localization + instruction

2. [Reflexion Paper (Published: 2023)](https://arxiv.org/abs/2303.11366)
   - Verbal reinforcement stored in episodic memory
   - Forces explicit citations and enumeration

3. [Anthropic Introspection Research (Published: January 2025)](https://www.anthropic.com/news/probes)
   - Model must have internal state recognition BEFORE verbalizing
   - Need to verify CONTENT, not just existence

**Solution Implemented**: RIC v5.1 Quality Gates
- Upgrade ideas must have (i) LOCATION (file:line) + (ii) ACTION verb
- Iteration 2+ must cite previous iteration (Reflexion pattern)
- Multi-metric validation instead of boolean flags

**Full Research Document**: [RIC-V51-QUALITY-GATES-RESEARCH.md](RIC-V51-QUALITY-GATES-RESEARCH.md)


---

### Research Entry - December 04, 2025 at 11:05 PM

**Search Queries**:
- "SELF-REFINE paper iterative output critique mechanism how it works"
- "Reflexion framework LLM agent implementation feedback loop architecture"
- "automated testing AI agents quality gates continuous verification 2025"

**Search Date**: December 04, 2025 at 11:05 PM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 04, 2025 at 11:06 PM

**Search Queries**:
- "LLM agent planning phase best practices task decomposition 2025"
- "AI code generation iterative development atomic commits verification 2025"

**URLs Discovered**:
- https://arxiv.org/html/2506.11442v1

**Search Date**: December 04, 2025 at 11:06 PM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 04, 2025 at 11:20 PM

**Search Queries**:
- "LangGraph CrewAI agent task planning SMART criteria success metrics 2025"
- "autonomous AI agent coordination frameworks OpenAI Anthropic"
- "Claude Code multi-agent orchestration patterns 2025"

**Search Date**: December 04, 2025 at 11:20 PM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 04, 2025 at 11:20 PM

**Search Queries**:
- "LLM multi-agent systems architecture best practices 2025"
- "claude-flow agent orchestration swarm intelligence MCP protocol"
- "agent memory persistence context sharing multi-agent systems"

**Search Date**: December 04, 2025 at 11:20 PM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 04, 2025 at 11:21 PM

**Search Queries**:
- "LangGraph agent orchestration workflow patterns state management"
- ""wshobson/agents" Claude Code 85 agents multi-agent plugins"
- "agent handoff patterns error recovery multi-agent fallback strategies"

**Search Date**: December 04, 2025 at 11:21 PM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 05, 2025 at 12:07 AM

**Search Queries**:
- "technical implementation checklist template engineering"
- "software upgrade guide template best practices 2025"
- "ADR architecture decision record template format"

**Search Date**: December 05, 2025 at 12:07 AM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 05, 2025 at 12:08 AM

**Search Queries**:
- "technical specification document template PRD product requirements"
- "project risk assessment matrix template RAID log"
- "RFC request for comments document template software engineering"

**Search Date**: December 05, 2025 at 12:08 AM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 05, 2025 at 12:09 AM

**Search Queries**:
- "DORA metrics DevOps tracking software delivery performance"
- ""definition of done" checklist software development template"
- "software design document sections components best structure"

**Search Date**: December 05, 2025 at 12:09 AM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 05, 2025 at 12:17 AM

**Search Queries**:
- "AI project documentation template machine learning ML experiment tracking"
- "technical guide tutorial documentation template best practices"
- "research documentation template best practices technical writing 2025"

**Search Date**: December 05, 2025 at 12:17 AM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 05, 2025 at 12:17 AM

**Search Queries**:
- "insight discovery documentation template product development"
- "user research findings documentation template UX"
- "lessons learned documentation template project management"

**Search Date**: December 05, 2025 at 12:17 AM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 05, 2025 at 12:37 AM

**Search Queries**:
- "knowledge base article template structure best practices"
- "software release notes template structured format machine readable 2025"
- "changelog upgrade documentation template best practices AI agents 2025"

**Search Date**: December 05, 2025 at 12:37 AM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 05, 2025 at 01:54 AM

**Search Queries**:
- "implementation checklist template agile software development task tracking 2025"
- "QuantConnect LEAN local deployment Docker API REST 2025"
- "thinkorswim UI architecture components design trading platform 2024 2025"

**Search Date**: December 05, 2025 at 01:54 AM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 05, 2025 at 01:55 AM

**Search Queries**:
- "QuantConnect MCP server Model Context Protocol Claude integration 2025"
- "Python trading dashboard PySide6 PyQt real-time charts candlestick 2025"
- "thinkorswim thinkScript custom studies indicators programming"

**Search Date**: December 05, 2025 at 01:55 AM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 05, 2025 at 01:55 AM

**Search Queries**:
- "autonomous AI agent trading bot dashboard UI real-time visualization 2024 2025"
- "QuantConnect LEAN streaming API websocket real-time data messaging 2025"
- "TradingView lightweight charts Python embed custom indicators overlay"

**Search Date**: December 05, 2025 at 01:55 AM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 05, 2025 at 01:56 AM

**Search Queries**:
- "trading UI components option chain grid position management order entry 2025"
- "Dash Plotly trading dashboard real-time streaming financial visualization 2025"
- "LEAN algorithm backtesting results API JSON output metrics performance"

**Search Date**: December 05, 2025 at 01:56 AM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.


---

### Research Entry - December 05, 2025 at 01:56 AM

**Search Queries**:
- "thinkorswim workspace layout customization dockable panels flex grid"
- "PySide6 dockable widgets QDockWidget layout manager trading application"
- "AI agent observability monitoring dashboard LLM decisions audit trail visualization"

**Search Date**: December 05, 2025 at 01:56 AM
**Note**: This entry was auto-persisted. Add publication dates and detailed analysis.
