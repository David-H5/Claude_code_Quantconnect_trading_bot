# Trading UI Research - thinkorswim-Style Dashboard for Autonomous AI Trading Bot

## Research Overview

**Date**: December 5, 2025
**Scope**: UI architecture, frameworks, and integration patterns for building a thinkorswim-style trading dashboard
**Focus**: QuantConnect LEAN integration, AI agent observability, Python UI frameworks
**Result**: Comprehensive research with actionable implementation recommendations

---

## Research Objectives

1. Understand thinkorswim UI architecture and key components
2. Explore QuantConnect LEAN local deployment and API options
3. Investigate QuantConnect MCP server integration with Claude
4. Research autonomous AI agent trading bot UI patterns
5. Evaluate Python UI frameworks for trading applications
6. Design component hierarchy for implementation

---

## Research Phases

### Phase 1: thinkorswim UI Architecture

**Search Date**: December 5, 2025 at 10:30 AM EST
**Search Queries**:
- "thinkorswim UI architecture components design trading platform 2024 2025"
- "thinkorswim thinkScript custom studies indicators programming"
- "thinkorswim workspace layout customization dockable panels flex grid"

**Key Sources**:

1. [thinkorswim Desktop - Charles Schwab (Published: 2024)](https://www.schwab.com/trading/thinkorswim/desktop)
2. [Flexible Grid - thinkorswim Learning Center (Updated: 2024)](https://toslc.thinkorswim.com/center/howToTos/thinkManual/charts/Flexible-Grid)
3. [thinkScript Reference (Updated: 2024)](https://toslc.thinkorswim.com/center/reference/thinkScript)
4. [Workspaces Guide (Updated: 2024)](https://toslc.thinkorswim.com/center/howToTos/thinkManual/Getting-Started/Workspaces)
5. [New thinkorswim Reskin - Ticker Tape (Published: 2023)](https://tickertape.tdameritrade.com/tools/new-thinkorswim-reskin-15344)
6. [Trade Architect Design Case Study (Published: ~2023)](https://enriquesallent.com/trade-architect)
7. [useThinkScript Community (Active: 2025)](https://usethinkscript.com/)

**Key Discoveries**:

#### UI Architecture Evolution
- thinkorswim underwent complete front-end code refactoring - all UI code was reviewed and updated
- Design Thinking methodology was applied for Trade Architect (web companion)
- Version 2.0 focused on accessibility - components normalized across the application
- Entry-level users found thinkorswim overwhelming, leading to simpler preset workspaces

#### Flexible Grid System
- Core UI pattern: Flexible Grid allows cells to be added, resized, and customized
- Cells can be added below (half height) or to right (half width) of existing cells
- Drag-and-drop borders resize cells proportionally
- Each cell can host different gadgets (charts, watchlists, options chains, etc.)
- Detachable windows allow multi-monitor setups

#### Workspace Management
- Multiple named workspaces can be saved and loaded
- Six pre-built workspace presets (Active Trader, Options, Futures, etc.)
- Workspaces stored locally (NOT on servers) - important for backup strategy
- Save via Setup > "Save Workspace as..."
- Remembers exact position, size, and configuration across all monitors

#### thinkScript Capabilities
- Built-in programming language for custom indicators and strategies
- No special programming knowledge required for basic usage
- Key commands: `reference` (pull existing studies), `plot` (display data)
- Can combine multiple indicators in single script
- Strategies use `AddOrder` function for backtesting

**Applied**: Workspace save/load system, Flexible Grid concept, detachable panels design

---

### Phase 2: QuantConnect LEAN Local Deployment

**Search Date**: December 5, 2025 at 10:35 AM EST
**Search Queries**:
- "QuantConnect LEAN local deployment Docker API REST 2025"
- "LEAN algorithm backtesting results API JSON output metrics performance"
- "QuantConnect LEAN streaming API websocket real-time data messaging 2025"

**Key Sources**:

1. [LEAN CLI - lean backtest (Updated: 2025)](https://www.quantconnect.com/docs/v2/lean-cli/api-reference/lean-backtest)
2. [LEAN Docker Image (Updated: 2025)](https://hub.docker.com/r/quantconnect/lean)
3. [lean-cli GitHub Repository (Active: 2025)](https://github.com/QuantConnect/lean-cli)
4. [Data Providers Documentation (Updated: 2025)](https://www.quantconnect.com/docs/v2/writing-algorithms/live-trading/data-providers)
5. [Results Documentation (Updated: 2025)](https://www.quantconnect.com/docs/v2/cloud-platform/backtesting/results)
6. [LEAN Reports Documentation (Updated: 2025)](https://www.lean.io/docs/v2/lean-cli/reports)
7. [Polygon.io Integration Blog (Published: 2024)](https://polygon.io/blog/integration-quantconnect)

**Key Discoveries**:

#### Docker-Based Architecture
- LEAN runs in `quantconnect/lean` Docker image containing all libraries
- Results stored in `<project>/backtest/<timestamp>` directory
- Custom output directory via `--output` option
- Networking: Use `host.docker.internal` instead of `localhost` for local services
- Custom Docker images can be built for modified LEAN versions

#### Backtest Results JSON Structure
- `completed` boolean indicates backtest status
- `statistics` contains performance metrics (Alpha, Beta, Sortino)
- `runtimeStatistics` for live updates during algorithm execution
- Rolling window statistics available
- Can download insights in JSON format

#### Live Data Streaming
- QuantConnect doesn't stream data directly (vendor restrictions)
- Real-time equity data requires Interactive Brokers or IQFeed
- WebSocket support via `BaseWebsocketsBrokerage` class
- Polygon.io integration supports tick-by-tick to daily data
- CoinAPI supports REST, WebSocket, and FIX protocols for crypto

#### Research Environment
- `lean research` command runs Jupyter Lab in Docker
- Project directory mounted in container
- `api` variable pre-configured for authenticated requests
- Warm Up feature prepares indicators for live trading

**Applied**: LEAN Bridge architecture, Docker integration pattern, WebSocket data streaming

---

### Phase 3: QuantConnect MCP Server Integration

**Search Date**: December 5, 2025 at 10:40 AM EST
**Search Queries**:
- "QuantConnect MCP server Model Context Protocol Claude integration 2025"

**Key Sources**:

1. [MCP Server Documentation (Published: 2025)](https://www.quantconnect.com/docs/v2/ai-assistance/mcp-server)
2. [MCP Server Landing Page (Published: 2025)](https://www.quantconnect.com/mcp)
3. [MCP Server GitHub Repository (Published: 2025)](https://github.com/QuantConnect/mcp-server)
4. [Claude Code MCP Integration (Published: 2025)](https://www.quantconnect.com/docs/v2/ai-assistance/mcp-server/claude-code)
5. [Model Context Protocol - Anthropic (Published: 2024)](https://www.anthropic.com/news/model-context-protocol)
6. [MCP Servers Repository (Active: 2025)](https://github.com/modelcontextprotocol/servers)

**Key Discoveries**:

#### MCP Server Capabilities
- Bridge connecting LLMs (Claude, GPT) to QuantConnect API
- 60+ API endpoints fully implemented and tested
- Covers: file/project management, coding, compiling, backtests, optimization, live trading
- Vector search of documentation, QCAlgorithm API, and thousands of examples
- Static API syntax analysis for quick error identification

#### Integration Patterns
- Requires Docker Desktop for deployment
- Pull image: `docker pull quantconnect/mcp-server`
- Configure via Claude Desktop settings or `.mcp.json`
- Supports both Claude Desktop (Pro $20/mo) and Claude Code
- Multi-platform: linux/amd64 (Intel/AMD), linux/arm64 (Apple M-series)

#### Local Platform Integration
- Can edit project files locally
- Deploy backtests from local environment
- Deploy to paper trading
- Works alongside LEAN CLI

#### Open Source Details
- Apache 2.0 license
- Complete test suite included
- CI/CD system for automatic Docker image builds
- Listed in official MCP servers repository

**Applied**: MCP integration for AI-assisted analysis, Claude Code workflow

---

### Phase 4: Autonomous AI Agent Trading Bot UI Patterns

**Search Date**: December 5, 2025 at 10:45 AM EST
**Search Queries**:
- "autonomous AI agent trading bot dashboard UI real-time visualization 2024 2025"
- "AI agent observability monitoring dashboard LLM decisions audit trail visualization"

**Key Sources**:

1. [FlowHunt AI Trading Bot (Published: 2025)](https://www.flowhunt.io/blog/flowhunt-ai-trading-bot-goes-live/)
2. [Best AI Trading Bots 2025 - StockBrokers.com (Published: 2025)](https://www.stockbrokers.com/guides/ai-stock-trading-bots)
3. [Crypto AI Agents Future 2025 - Codezeros (Published: 2025)](https://www.codezeros.com/what-are-crypto-ai-agents-the-future-of-autonomous-trading-in-2025)
4. [AI Trading Guide - WunderTrading (Published: 2025)](https://wundertrading.com/journal/en/learn/article/guide-to-ai-trading-bots)
5. [Datadog LLM Observability (Active: 2025)](https://www.datadoghq.com/product/llm-observability/)
6. [AgentOps (Active: 2025)](https://www.agentops.ai/)
7. [Arize AI Observability (Active: 2025)](https://arize.com/)
8. [Dynatrace AI Observability (Active: 2025)](https://www.dynatrace.com/solutions/ai-observability/)
9. [AI Audit Trail - Medium Article (Published: Oct 2025)](https://medium.com/@kuldeep.paul08/the-ai-audit-trail-how-to-ensure-compliance-and-transparency-with-llm-observability-74fd5f1968ef)
10. [Datadog AI Agents Monitoring (Published: 2025)](https://www.datadoghq.com/blog/monitor-ai-agents/)

**Key Discoveries**:

#### AI Trading Dashboard Requirements
- Real-time dashboard with instant visibility into bot performance
- Performance visualization across different time periods and market conditions
- Next.js + TypeScript popular for responsive dashboards
- React components for real-time data visualization
- Public visibility of trades for transparency (FlowHunt pattern)

#### LLM Observability Patterns
- End-to-end tracing across AI agent workflows
- Visibility into inputs, outputs, latency, token usage, errors
- Graph-based view of entire decision process
- Multi-agent workflow visualization (handoffs, retries, errors)
- Tools for each step in agent workflow

#### Audit Trail Requirements
- Track every input/output for compliance
- Full data lineage from prompt to response
- Query data in real-time, store for future reference
- Support for regulatory standards (documented in CLAUDE.md)

#### Decision Visualization Challenges
- Agentic systems are nonlinear (parallel activity, non-deterministic)
- Traditional visualizations inadequate for complex fan-in/fan-out patterns
- Need specialized graph-based views

#### Key Platforms
- **Datadog**: End-to-end tracing, multi-agent workflow visualization
- **AgentOps**: Visual event tracking, replay capabilities
- **Arize**: OpenTelemetry-based, heatmaps for failure modes
- **Weights & Biases Weave**: Multi-agent LLM monitoring, hierarchical call tracking
- **Langfuse/Langtrace**: Granular tracing, cost tracking

#### Market Trends
- Grand View estimates autonomous fintech software > $52.4B by 2026
- Companies like 3Commas, Alpaca, WunderTrading deploying advanced products
- LLM integration for natural language strategy description (Gunbot pattern)

**Applied**: Decision timeline UI, audit trail panel, agent metrics visualization, debate viewer

---

### Phase 5: Python Trading UI Frameworks

**Search Date**: December 5, 2025 at 10:50 AM EST
**Search Queries**:
- "Python trading dashboard PySide6 PyQt real-time charts candlestick 2025"
- "TradingView lightweight charts Python embed custom indicators overlay"
- "Dash Plotly trading dashboard real-time streaming financial visualization 2025"
- "PySide6 dockable widgets QDockWidget layout manager trading application"

**Key Sources**:

1. [lightweight-charts-python - GitHub (Active: 2025)](https://github.com/louisnw01/lightweight-charts-python)
2. [lightweight-charts-python Documentation (Active: 2025)](https://lightweight-charts-python.readthedocs.io/)
3. [PySide6 QDockWidget Documentation (Updated: 2024)](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QDockWidget.html)
4. [Qt-Advanced-Docking-System (Active: 2025)](https://github.com/githubuser0xFFFF/Qt-Advanced-Docking-System)
5. [PyQtGraph with PySide6 (Active: 2025)](https://www.pythonguis.com/tutorials/pyside6-plotting-pyqtgraph/)
6. [QCandlestickSeries - Qt for Python (Updated: 2024)](https://doc.qt.io/qtforpython-6/PySide6/QtCharts/QCandlestickSeries.html)
7. [Plotly Dash Finance Examples (Active: 2025)](https://plotly.com/examples/finance/)
8. [Real-Time Stock Dashboard with Plotly (Published: 2025)](https://community.plotly.com/t/an-indian-real-time-stock-market-dashboard-with-comprehensive-financial-analytics-created-using-plotly-dash/92877)
9. [Interactive Dash Dashboard Guide (Published: Sept 2025)](https://www.marktechpost.com/how-to-design-an-interactive-dash-and-plotly-dashboard-with-callback-mechanisms-for-local-and-online-deployment/)
10. [TradingView Lightweight Charts - Official (Active: 2025)](https://www.tradingview.com/lightweight-charts/)

**Key Discoveries**:

#### lightweight-charts-python
- TradingView-quality charts in Python
- Supports: Jupyter, PyQt5/6, PySide6, wxPython, Streamlit, asyncio
- Features: Live data streaming, multi-pane subcharts, Toolbox with drawing tools
- `create_line()` for indicator overlays
- Fork `lightweight-charts-esistjosh` includes many pre-built indicators
- Important: Must implement indicators yourself (no TradingView indicator library)
- Installation: `pip install lightweight-charts`

#### PySide6 + QDockWidget
- Native Qt docking system for utility windows
- Dock areas: left, right, top, bottom around central widget
- Features: move between areas, float as top-level windows
- `setDockNestingEnabled()` for multi-direction docking
- `saveState()`/`restoreState()` for layout persistence
- Child widget controls size hints and policies

#### Qt-Advanced-Docking-System
- VS Code-style docking for complex applications
- Complete Python bindings (PyQt5, PyQt6, PySide6)
- Available via PyPI (`PyQtAds`)
- Cross-platform (Windows, macOS, Linux)
- Solves Qt's basic docking limitations

#### Multiple Central Widgets Pattern
- Set `QMainWindow` as central widget with only dock widgets
- Limitation: Central dock-widgets can't interchange with outer
- Need menu for restoring closed dock widgets

#### Dash/Plotly for Trading
- Used by S&P Global, Liberty Mutual, Santander
- Flask backend with local server
- Callbacks for real-time responsiveness
- Challenge: Graph updates can be choppy (new graph objects per callback)
- Best for analysis dashboards, less ideal for high-frequency trading UI

#### Performance Considerations
- Matplotlib slow for thousands of candles
- PyQt/PySide faster and more visually appealing
- PyQtGraph built on Qt's native QGraphicsScene for live data performance
- Virtual scrolling needed for large datasets

**Applied**: PySide6 + Qt-ADS selection, lightweight-charts-python for charts, docking architecture

---

### Phase 6: Trading UI Components

**Search Date**: December 5, 2025 at 10:55 AM EST
**Search Queries**:
- "trading UI components option chain grid position management order entry 2025"

**Key Sources**:

1. [Interactive Brokers Option Chain Guide (Active: 2025)](https://www.ibkrguides.com/traderworkstation/option-chain.htm)
2. [Trading Technologies Market Grid (Active: 2025)](https://library.tradingtechnologies.com/trade/mg-trading-from-market-grid.html)
3. [TraderEvolution Option Chain (Active: 2025)](https://guide.traderevolution.com/project/web-platform/trading-panels/option-master/option-chain)
4. [OptionX Price Ladder Guide (Published: 2025)](https://optionx.trade/support/getting-started/introduction/ultimate-guide-to-price-ladder-trading-with-optionx)
5. [CME Group Options Grid (Active: 2025)](https://www.cmegroup.com/tools-information/webhelp/cmeone-cmed-mobile/Content/OptionsGrid.html)

**Key Discoveries**:

#### Option Chain UI Pattern
- Spread Template feature for comparing similar spreads
- Click tile to load Strategy Builder with desired spread
- Order Entry panel for parameters + Submit button
- Strike-centered view with calls/puts on sides
- Edit 'Trade' column to add positions (negative = short, positive = long)

#### DOM/Price Ladder Features
- One-click order execution directly on ladder
- Visual order management (drag-and-drop modification)
- Strategy visualization across all legs
- Linked execution for bracket/cover orders
- Market depth display

#### Options Grid Components
- ATM-centered display based on underlying + expiration
- View market activity and summary data
- Aggress and submit orders from grid
- Filter by strike range, expiration, etc.

#### Order Management Panel
- Tabs: Orders (open), Grids (order groups)
- Grid = group of orders with control functions
- Can apply control functions to entire grid

**Applied**: Options chain grid design, DOM ladder component, order management panel

---

## Critical Discoveries Summary

### 1. Framework Selection: PySide6 + Qt-ADS

**Rationale**:
- PySide6 provides professional-grade native UI toolkit
- Qt-Advanced-Docking-System enables VS Code-style flexibility
- lightweight-charts-python integrates seamlessly with PySide6
- Better performance than web-based alternatives for real-time data

### 2. Chart Implementation: lightweight-charts-python

**Rationale**:
- TradingView-quality rendering
- Native PySide6 integration
- Supports live streaming updates
- Toolbox with drawing tools included
- Must implement indicators manually (our existing indicator modules)

### 3. LEAN Integration: Docker + WebSocket Bridge

**Rationale**:
- LEAN runs in Docker for consistency
- WebSocket provides real-time data streaming
- REST API for control operations
- MCP server for AI-assisted analysis

### 4. AI Agent Observability: Custom Dashboard

**Rationale**:
- Integrate existing decision_logger.py infrastructure
- Build custom visualization for trading-specific needs
- OpenTelemetry-compatible patterns from industry leaders
- Specialized debate viewer for Bull/Bear mechanism

### 5. Workspace Management: JSON-Based Persistence

**Rationale**:
- Qt's saveState()/restoreState() for dock positions
- Custom JSON for panel configurations
- Named workspaces like thinkorswim
- Export/import for backup (critical - not stored on servers)

---

## Research Deliverables

| Deliverable | Location | Lines |
|-------------|----------|-------|
| UPGRADE-019 Implementation Guide | docs/research/UPGRADE-019-TRADING-UI-THINKORSWIM-STYLE.md | ~600 |
| This Research Document | docs/research/TRADING-UI-THINKORSWIM-RESEARCH-DEC2025.md | ~500 |
| Insight Document | docs/insights/INSIGHT-2025-12-05-TRADING-UI-ARCHITECTURE.md | ~300 |

---

## Changelog

| Date | Change | Impact |
|------|--------|--------|
| 2025-12-05 | Initial research completed | Foundation for UPGRADE-019 |
| 2025-12-05 | Created UPGRADE document | Implementation roadmap |
| 2025-12-05 | Created research documentation | Comprehensive reference |
