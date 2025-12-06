# UPGRADE-019: Interactive Trading UI (thinkorswim-Style)

## Overview

**Date Created**: December 5, 2025
**Status**: PLANNING
**Priority**: P1 - High
**Estimated Effort**: Large (4-6 weeks implementation)

### Objective

Build an interactive trading dashboard for the autonomous AI agent trading bot that:
1. Integrates with local QuantConnect LEAN engine
2. Provides thinkorswim-style interactivity and appearance
3. Displays real-time AI agent decision-making and audit trails
4. Supports multi-pane customizable layouts with dockable panels

### Key Deliverables

| Deliverable | Description | Priority |
|-------------|-------------|----------|
| Core Dashboard Framework | PySide6-based main window with docking | P0 |
| Chart System | TradingView-style charts with indicators | P0 |
| LEAN Integration | WebSocket/REST bridge to local LEAN | P0 |
| Options Chain Panel | Interactive options grid with Greeks | P1 |
| AI Agent Monitor | Decision visualization and audit trail | P1 |
| Order Entry Panel | Quick order entry with DOM ladder | P1 |
| Position Management | Real-time P&L and position tracking | P1 |
| Workspace Manager | Save/load custom layouts | P2 |
| thinkScript-like DSL | Custom indicator scripting | P3 |

---

## Phase 1: Foundation (Week 1-2)

### Task 1.1: Core Dashboard Framework

**Status**: [ ] Not Started

Create the main application window using PySide6 with advanced docking capabilities.

**Technical Approach**:
- Use `Qt-Advanced-Docking-System` for VS Code-style docking
- Implement `QMainWindow` with nested dock areas
- Support drag-and-drop panel rearrangement
- Save/restore workspace layouts to JSON

**Files to Create**:
```
ui/
├── dashboard/
│   ├── __init__.py
│   ├── main_window.py          # QMainWindow with docking
│   ├── dock_manager.py         # Advanced docking controller
│   ├── workspace_manager.py    # Layout save/load
│   └── theme_manager.py        # Dark/light themes
```

**Key Components**:
```python
# main_window.py structure
class TradingDashboard(QMainWindow):
    """Main trading dashboard with thinkorswim-style docking."""

    def __init__(self):
        # Initialize Qt-ADS dock manager
        # Create central widget (chart area)
        # Initialize dock areas (left, right, bottom)
        # Load default workspace

    def add_dock_widget(self, widget, area, title):
        """Add dockable panel to specified area."""

    def save_workspace(self, name: str):
        """Save current layout to workspace file."""

    def load_workspace(self, name: str):
        """Restore layout from workspace file."""
```

**Dependencies**:
- `PySide6>=6.6.0`
- `PyQtAds>=4.3.0` (Qt-Advanced-Docking-System)

### Task 1.2: Theme System (thinkorswim Dark Theme)

**Status**: [ ] Not Started

Implement thinkorswim-style dark theme with professional trading aesthetics.

**Color Palette**:
```python
TOS_DARK_THEME = {
    "background": "#1E1E1E",
    "panel_bg": "#252525",
    "border": "#3E3E3E",
    "text_primary": "#FFFFFF",
    "text_secondary": "#B0B0B0",
    "accent": "#00A0FF",
    "profit_green": "#00C853",
    "loss_red": "#FF1744",
    "warning_yellow": "#FFD600",
    "chart_grid": "#2A2A2A",
}
```

**Files to Create**:
```
ui/themes/
├── __init__.py
├── dark_theme.qss          # Qt stylesheet
├── light_theme.qss
└── theme_manager.py
```

### Task 1.3: LEAN Integration Bridge

**Status**: [ ] Not Started

Create communication bridge between UI and local LEAN engine.

**Architecture**:
```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│  Trading UI     │◄──────────────────►│  LEAN Bridge    │
│  (PySide6)      │                    │  (FastAPI)      │
└─────────────────┘                    └────────┬────────┘
                                                │
                                       ┌────────▼────────┐
                                       │  LEAN Engine    │
                                       │  (Docker)       │
                                       └─────────────────┘
```

**Bridge Components**:
```python
# api/lean_bridge.py
class LEANBridge:
    """Bridge for communicating with local LEAN engine."""

    async def start_backtest(self, algorithm: str, params: dict) -> str:
        """Start backtest and return job ID."""

    async def get_backtest_results(self, job_id: str) -> BacktestResult:
        """Get backtest results JSON."""

    async def stream_live_data(self, symbols: list) -> AsyncIterator[Quote]:
        """Stream real-time quotes from LEAN."""

    async def submit_order(self, order: Order) -> OrderResult:
        """Submit order to LEAN."""
```

**Files to Create**:
```
api/
├── __init__.py
├── lean_bridge.py           # LEAN communication
├── websocket_server.py      # WebSocket for UI
├── rest_server.py           # REST API for control
└── data_models.py           # Pydantic models
```

---

## Phase 2: Chart System (Week 2-3)

### Task 2.1: TradingView-Style Charts

**Status**: [ ] Not Started

Implement professional trading charts using `lightweight-charts-python`.

**Features**:
- Candlestick, line, area, and bar charts
- Multiple timeframes (1m, 5m, 15m, 1h, 4h, 1D, 1W)
- Crosshair with price/time display
- Zoom and pan navigation
- Volume bars
- Drawing tools (trendlines, rectangles, Fibonacci)

**Technical Approach**:
```python
# ui/charts/chart_widget.py
from lightweight_charts import Chart

class TradingChart(QWidget):
    """TradingView-style chart widget using lightweight-charts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.chart = Chart(toolbox=True)

    def set_data(self, ohlcv: pd.DataFrame):
        """Set OHLCV data for chart."""
        self.chart.set(ohlcv)

    def add_indicator(self, indicator: Indicator):
        """Add technical indicator overlay."""
        line = self.chart.create_line()
        line.set(indicator.values)

    def add_volume(self, volume: pd.Series):
        """Add volume bars subchart."""
        self.chart.volume_config(up_color='#00C853', down_color='#FF1744')
```

**Files to Create**:
```
ui/charts/
├── __init__.py
├── chart_widget.py          # Main chart component
├── chart_toolbar.py         # Timeframe/indicator controls
├── indicator_panel.py       # Indicator configuration
├── drawing_tools.py         # Trendlines, Fibonacci, etc.
└── subchart_manager.py      # Volume, RSI subcharts
```

### Task 2.2: Technical Indicators Integration

**Status**: [ ] Not Started

Integrate existing indicators from `indicators/` module with chart overlays.

**Supported Indicators**:
| Indicator | Type | Overlay |
|-----------|------|---------|
| SMA | Moving Average | Yes |
| EMA | Moving Average | Yes |
| VWAP | Volume-weighted | Yes |
| Bollinger Bands | Volatility | Yes |
| Keltner Channels | Volatility | Yes |
| RSI | Momentum | Subchart |
| MACD | Momentum | Subchart |
| CCI | Momentum | Subchart |
| OBV | Volume | Subchart |
| Ichimoku Cloud | Trend | Yes |

**Integration Pattern**:
```python
# Bridge existing indicators to chart
from indicators import VWAP, BollingerBands, RSI

class IndicatorManager:
    def add_vwap(self, chart: TradingChart, data: pd.DataFrame):
        vwap = VWAP().calculate(data)
        chart.add_overlay(vwap, color='#FFD600', name='VWAP')
```

---

## Phase 3: Options Chain Panel (Week 3-4)

### Task 3.1: Options Chain Grid

**Status**: [ ] Not Started

Build interactive options chain grid similar to thinkorswim.

**Features**:
- Strike-centered view with calls on left, puts on right
- Real-time Greeks (Delta, Gamma, Theta, Vega, Rho)
- Implied Volatility column
- Open Interest and Volume
- Bid/Ask with spread highlighting
- Expiration date selector
- One-click trade from grid

**UI Layout**:
```
┌─────────────────────────────────────────────────────────────────┐
│ Expiration: [Dec 20 ▼]  Strikes: [±10 ▼]  Filter: [All ▼]       │
├─────────────────────────┬───────┬─────────────────────────────────┤
│         CALLS           │Strike │           PUTS                  │
├────┬────┬────┬────┬─────┼───────┼─────┬────┬────┬────┬─────┬──────┤
│Bid │Ask │ Δ  │ IV │ OI  │       │ OI  │ IV │ Δ  │Bid │Ask  │      │
├────┼────┼────┼────┼─────┼───────┼─────┼────┼────┼────┼─────┼──────┤
│4.50│4.55│0.45│32% │5.2K │  450  │3.1K │35% │-.42│3.20│3.25 │      │
│3.80│3.85│0.38│30% │8.1K │  455  │4.5K │33% │-.38│2.75│2.80 │      │
│3.15│3.20│0.32│29% │12K  │  460  │6.2K │31% │-.32│2.35│2.40 │  ATM │
│2.55│2.60│0.26│28% │9.3K │  465  │5.8K │30% │-.26│1.95│2.00 │      │
└────┴────┴────┴────┴─────┴───────┴─────┴────┴────┴────┴─────┴──────┘
```

**Files to Create**:
```
ui/options/
├── __init__.py
├── chain_widget.py          # Main options chain grid
├── chain_model.py           # Qt model for chain data
├── greeks_calculator.py     # Real-time Greeks display
├── expiration_selector.py   # Expiration dropdown
└── strategy_builder.py      # Multi-leg strategy UI
```

### Task 3.2: Strategy Builder (Multi-Leg)

**Status**: [ ] Not Started

Build visual strategy builder for multi-leg options positions.

**Supported Strategies** (via QuantConnect OptionStrategies):
- Vertical Spreads (Bull Call, Bear Put)
- Iron Condor
- Iron Butterfly
- Butterfly (Call/Put)
- Straddle/Strangle
- Covered Call
- Collar

**UI Pattern**:
```python
class StrategyBuilder(QWidget):
    """Visual multi-leg options strategy builder."""

    def add_leg(self, contract: OptionContract, quantity: int, side: str):
        """Add leg to strategy."""

    def calculate_payoff(self) -> PayoffDiagram:
        """Calculate and display P&L payoff diagram."""

    def submit_as_combo(self):
        """Submit as ComboLimitOrder to LEAN."""
```

---

## Phase 4: AI Agent Monitor (Week 4-5)

### Task 4.1: Agent Decision Dashboard

**Status**: [ ] Not Started

Build real-time visualization of AI agent decisions and reasoning.

**Features**:
- Decision timeline with reasoning chains
- Confidence scores visualization
- Bull/Bear debate viewer (from existing `debate_mechanism.py`)
- Agent performance metrics
- Decision audit trail

**UI Layout**:
```
┌─────────────────────────────────────────────────────────────────┐
│ AI Agent Monitor                                    [Pause] [↻] │
├───────────────────────────────────┬─────────────────────────────┤
│ Decision Timeline                 │ Current Analysis            │
│ ┌───────────────────────────────┐ │ ┌─────────────────────────┐ │
│ │ 14:32:15 BUY SPY 450C         │ │ │ Symbol: SPY             │ │
│ │   Confidence: 85%             │ │ │ Signal: BULLISH         │ │
│ │   Reasoning: [Expand ▼]       │ │ │ Confidence: 85%         │ │
│ │                               │ │ │                         │ │
│ │ 14:28:42 HOLD AAPL           │ │ │ Reasoning Steps:        │ │
│ │   Confidence: 62%             │ │ │ 1. RSI oversold (28)    │ │
│ │   Debate: No consensus        │ │ │ 2. IV below 25th pctl   │ │
│ │                               │ │ │ 3. News sentiment +0.3  │ │
│ └───────────────────────────────┘ │ └─────────────────────────┘ │
├───────────────────────────────────┼─────────────────────────────┤
│ Agent Metrics                     │ Bull/Bear Debate            │
│ ┌───────────────────────────────┐ │ ┌─────────────────────────┐ │
│ │ Accuracy:      78.5%          │ │ │ Round 1:                │ │
│ │ Calibration:   0.05           │ │ │ 🐂 85%: Strong support  │ │
│ │ Avg Confidence: 72%           │ │ │ 🐻 65%: Overbought...   │ │
│ │ Decisions/hr:  12             │ │ │                         │ │
│ └───────────────────────────────┘ │ │ Outcome: BUY (consensus)│ │
│                                   │ └─────────────────────────┘ │
└───────────────────────────────────┴─────────────────────────────┘
```

**Files to Create**:
```
ui/agents/
├── __init__.py
├── decision_timeline.py     # Scrolling decision history
├── reasoning_viewer.py      # Expandable reasoning chains
├── debate_viewer.py         # Bull/Bear debate visualization
├── metrics_panel.py         # Agent performance metrics
└── audit_trail.py           # Full decision audit log
```

### Task 4.2: LLM Observability Integration

**Status**: [ ] Not Started

Integrate with existing agent infrastructure for observability.

**Integration Points**:
- `llm/decision_logger.py` - Decision audit trail
- `evaluation/agent_metrics.py` - Performance metrics
- `llm/agents/debate_mechanism.py` - Debate visualization
- `llm/self_evolving_agent.py` - Evolution monitoring

**Data Flow**:
```python
# Connect existing infrastructure to UI
from llm.decision_logger import DecisionLogger
from evaluation.agent_metrics import AgentMetricsTracker

class AgentMonitorController:
    def __init__(self, logger: DecisionLogger, tracker: AgentMetricsTracker):
        self.logger = logger
        self.tracker = tracker

    def get_recent_decisions(self, limit: int = 50) -> List[DecisionLog]:
        """Get recent decisions for timeline."""
        return self.logger.get_recent_logs(limit)

    def get_metrics(self, agent_name: str) -> AgentMetrics:
        """Get metrics for specific agent."""
        return self.tracker.get_metrics(agent_name)
```

---

## Phase 5: Order Entry & Position Management (Week 5-6)

### Task 5.1: Quick Order Entry Panel

**Status**: [ ] Not Started

Build thinkorswim-style quick order entry widget.

**Features**:
- One-click buy/sell buttons
- Quantity spinner
- Order type selector (Market, Limit, Stop, Stop-Limit)
- Price input with tick buttons (±0.01, ±0.05, ±0.10)
- Time-in-force selector (DAY, GTC, IOC, FOK)
- Order preview before submission

**UI Layout**:
```
┌─────────────────────────────────────────┐
│ SPY @ 450.25                     [Chain]│
├─────────────────────────────────────────┤
│ Qty: [100 ▲▼]  Type: [Limit ▼]          │
│                                         │
│ Price: [450.20 ▲▼] [−.05][−.01][+.01][+.05]│
│                                         │
│ TIF: [DAY ▼]                            │
│                                         │
│ [    BUY    ]  [   SELL   ]             │
│     +450.20       -450.30               │
├─────────────────────────────────────────┤
│ Est. Cost: $45,020.00  |  Buying Power: │
│ Commission: $0.65      |  $127,450.00   │
└─────────────────────────────────────────┘
```

### Task 5.2: DOM/Price Ladder

**Status**: [ ] Not Started

Build Depth of Market (DOM) price ladder for quick trading.

**Features**:
- Vertical price ladder showing bid/ask depth
- One-click order placement at price levels
- Drag-and-drop order modification
- Working orders highlighted
- Imbalance indicators

**Files to Create**:
```
ui/trading/
├── __init__.py
├── order_entry.py           # Quick order entry widget
├── dom_ladder.py            # Price ladder/DOM
├── position_panel.py        # Position management
├── order_book.py            # Working orders list
└── execution_monitor.py     # Fill notifications
```

### Task 5.3: Position Management Panel

**Status**: [ ] Not Started

Real-time position tracking with P&L.

**Features**:
- Live P&L (unrealized and realized)
- Position Greeks (portfolio Delta, Gamma, Theta)
- Risk metrics (buying power, margin usage)
- One-click close/flatten buttons
- Position grouping by strategy

---

## Phase 6: Workspace Management (Week 6)

### Task 6.1: Layout Save/Load System

**Status**: [ ] Not Started

Implement thinkorswim-style workspace management.

**Features**:
- Multiple named workspaces
- Default presets (Options Trading, Day Trading, Analysis)
- Export/import workspace files
- Auto-save on exit

**Workspace JSON Structure**:
```json
{
  "name": "Options Trading",
  "version": "1.0",
  "panels": [
    {
      "type": "chart",
      "area": "center",
      "config": {
        "symbol": "SPY",
        "timeframe": "1D",
        "indicators": ["VWAP", "BB(20,2)"]
      }
    },
    {
      "type": "options_chain",
      "area": "right",
      "config": {
        "symbol": "SPY",
        "expiration": "nearest_monthly"
      }
    },
    {
      "type": "agent_monitor",
      "area": "bottom",
      "config": {
        "show_debate": true
      }
    }
  ],
  "geometry": {
    "width": 1920,
    "height": 1080,
    "dock_sizes": {...}
  }
}
```

### Task 6.2: Pre-built Workspace Presets

**Status**: [ ] Not Started

Create default workspace configurations.

| Workspace | Primary Focus | Key Panels |
|-----------|---------------|------------|
| Options Trading | Options analysis | Chart, Chain, Greeks, Strategy Builder |
| Day Trading | Quick execution | Chart, DOM, Order Entry, Positions |
| AI Monitor | Agent oversight | Decision Timeline, Metrics, Debate, Audit |
| Analysis | Research | Multi-chart, Watchlist, News, Scanner |
| Backtest | Strategy testing | Chart, Backtest Results, Metrics |

---

## Technical Architecture

### Component Hierarchy

```
TradingDashboard (QMainWindow)
├── MenuBar
│   ├── File (Workspaces, Export, Settings)
│   ├── View (Panels, Theme)
│   ├── Tools (Indicators, Drawing, Scanner)
│   └── Help
├── Toolbar
│   ├── Symbol Search
│   ├── Quick Actions
│   └── Connection Status
├── DockManager (Qt-ADS)
│   ├── CentralWidget (Chart Area)
│   │   └── ChartContainer
│   │       ├── TradingChart (lightweight-charts)
│   │       ├── ChartToolbar
│   │       └── IndicatorPanel
│   ├── LeftDock
│   │   ├── Watchlist
│   │   └── Scanner Results
│   ├── RightDock
│   │   ├── OptionsChain
│   │   ├── OrderEntry
│   │   └── PositionPanel
│   └── BottomDock
│       ├── AgentMonitor
│       ├── OrderBook
│       └── MessageLog
└── StatusBar
    ├── Connection Status
    ├── Account Balance
    └── Market Status
```

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Trading Dashboard                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Charts  │  │ Options │  │ Orders  │  │ Agents  │            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │                   │
│       └────────────┴─────┬──────┴────────────┘                   │
│                          │                                       │
│                 ┌────────▼────────┐                              │
│                 │  DataController │                              │
│                 │  (Qt Signals)   │                              │
│                 └────────┬────────┘                              │
└──────────────────────────┼───────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
     ┌────────▼────┐ ┌─────▼─────┐ ┌────▼────────┐
     │  WebSocket  │ │ REST API  │ │ LEAN Bridge │
     │  (quotes)   │ │ (control) │ │ (Docker)    │
     └─────────────┘ └───────────┘ └─────────────┘
```

### Technology Stack

| Component | Technology | Reason |
|-----------|------------|--------|
| UI Framework | PySide6 | Qt maturity, cross-platform, professional look |
| Docking | Qt-Advanced-Docking-System | VS Code-style flexibility |
| Charts | lightweight-charts-python | TradingView quality, fast rendering |
| Data Backend | FastAPI + WebSockets | Async, high-performance |
| LEAN Integration | Docker SDK + REST | Standard deployment pattern |
| State Management | Qt Signals/Slots | Native Qt pattern |
| Persistence | JSON + SQLite | Simple, portable |

---

## Integration with Existing Infrastructure

### Files to Integrate

| Existing File | Integration Point |
|---------------|-------------------|
| `ui/dashboard.py` | Extend or replace with new dashboard |
| `ui/widgets.py` | Reuse widget patterns |
| `llm/decision_logger.py` | Agent decision timeline |
| `evaluation/agent_metrics.py` | Metrics panel |
| `llm/agents/debate_mechanism.py` | Debate viewer |
| `execution/smart_execution.py` | Order submission |
| `scanners/options_scanner.py` | Scanner results panel |
| `indicators/` | Chart indicator overlays |
| `api/rest_server.py` | Extend with WebSocket |

### QuantConnect MCP Integration

The dashboard can leverage the QuantConnect MCP server for enhanced AI capabilities:

```python
# Use MCP for AI-assisted analysis
class MCPIntegration:
    async def analyze_with_claude(self, symbol: str, context: dict) -> Analysis:
        """Use Claude via MCP for market analysis."""
        # MCP server handles Claude communication
        # Returns structured analysis for UI display
```

---

## Dependencies to Add

```
# requirements-ui.txt
PySide6>=6.6.0
PyQtAds>=4.3.0              # Qt-Advanced-Docking-System
lightweight-charts>=2.0.0    # TradingView-style charts
fastapi>=0.109.0            # REST/WebSocket backend
uvicorn>=0.27.0             # ASGI server
websockets>=12.0            # WebSocket client
docker>=7.0.0               # LEAN Docker integration
aiohttp>=3.9.0              # Async HTTP client
```

---

## Testing Strategy

### Unit Tests

```
tests/ui/
├── test_chart_widget.py
├── test_options_chain.py
├── test_order_entry.py
├── test_workspace_manager.py
└── test_lean_bridge.py
```

### Integration Tests

- Full UI interaction with mock LEAN
- WebSocket streaming tests
- Order flow end-to-end

### Visual Regression Tests

- Screenshot comparison for UI consistency
- Theme rendering validation

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Qt-ADS complexity | Medium | Start with basic docking, iterate |
| LEAN Docker latency | Medium | Local WebSocket caching |
| Chart performance with large data | High | Virtual scrolling, data decimation |
| Cross-platform styling | Low | Qt stylesheets, consistent themes |
| MCP server availability | Low | Graceful fallback to local-only |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| UI responsiveness | <100ms interaction latency |
| Chart render time | <500ms for 10K candles |
| WebSocket latency | <50ms to LEAN |
| Memory usage | <500MB typical |
| Startup time | <5 seconds |

---

## References

### thinkorswim Documentation
- [thinkorswim Desktop](https://www.schwab.com/trading/thinkorswim/desktop)
- [Flexible Grid](https://toslc.thinkorswim.com/center/howToTos/thinkManual/charts/Flexible-Grid)
- [thinkScript Reference](https://toslc.thinkorswim.com/center/reference/thinkScript)
- [Workspaces](https://toslc.thinkorswim.com/center/howToTos/thinkManual/Getting-Started/Workspaces)

### QuantConnect/LEAN
- [LEAN CLI Documentation](https://www.quantconnect.com/docs/v2/lean-cli/api-reference/lean-backtest)
- [MCP Server Documentation](https://www.quantconnect.com/docs/v2/ai-assistance/mcp-server)
- [LEAN Docker Image](https://hub.docker.com/r/quantconnect/lean)

### UI Frameworks
- [Qt-Advanced-Docking-System](https://github.com/githubuser0xFFFF/Qt-Advanced-Docking-System)
- [lightweight-charts-python](https://github.com/louisnw01/lightweight-charts-python)
- [PySide6 Documentation](https://doc.qt.io/qtforpython-6/)

### AI Agent Observability
- [Datadog LLM Observability](https://www.datadoghq.com/product/llm-observability/)
- [AgentOps](https://www.agentops.ai/)
- [Arize AI Observability](https://arize.com/)

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2025-12-05 | Initial document creation | Claude |
