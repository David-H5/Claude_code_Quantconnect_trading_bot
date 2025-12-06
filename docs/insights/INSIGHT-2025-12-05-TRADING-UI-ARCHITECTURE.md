# Insight: Trading UI Architecture for Autonomous AI Trading Bot

## Metadata

**Date**: December 5, 2025
**Type**: Architecture Design Insight
**Confidence**: High (85%)
**Related**: UPGRADE-019, TRADING-UI-RESEARCH-DEC2025

---

## Executive Summary

After extensive research into thinkorswim UI patterns, QuantConnect LEAN integration, autonomous AI agent dashboards, and Python UI frameworks, the recommended architecture is:

**PySide6 + Qt-Advanced-Docking-System + lightweight-charts-python + FastAPI/WebSocket Bridge**

This combination provides professional-grade trading UI capabilities with the flexibility needed for AI agent observability while maintaining high performance for real-time data.

---

## Key Architectural Decisions

### Decision 1: PySide6 over Dash/Plotly/Web

**Choice**: Native desktop application using PySide6
**Alternatives Considered**: Dash/Plotly (web), Electron (hybrid), PyQt6
**Rationale**:
- **Performance**: Native Qt provides 60fps rendering for real-time data
- **Docking**: Qt-ADS enables VS Code-style panel management (Dash lacks this)
- **Integration**: Direct integration with lightweight-charts via QWidget embedding
- **Offline**: Works without network (critical for trading reliability)
- **Licensing**: PySide6 uses LGPL (more permissive than PyQt6 GPL)

**Trade-offs**:
- Requires installation (vs. browser access)
- More complex deployment than web
- Less natural for remote access (mitigated by potential future web dashboard)

### Decision 2: Qt-Advanced-Docking-System

**Choice**: PyQtAds (Qt-ADS) for docking
**Alternatives Considered**: Native QDockWidget, custom solution
**Rationale**:
- Native QDockWidget limited to 4 dock areas
- Qt-ADS provides IDE-like flexibility (tabs, splits, floating)
- Well-maintained with active Python bindings
- VS Code users will find familiar

**Implementation Pattern**:
```python
from PySide6.QtWidgets import QMainWindow
from ads import CDockManager, CDockWidget

class TradingDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dock_manager = CDockManager(self)

    def add_panel(self, widget, title, area):
        dock_widget = CDockWidget(title)
        dock_widget.setWidget(widget)
        self.dock_manager.addDockWidget(area, dock_widget)
```

### Decision 3: lightweight-charts-python for Charts

**Choice**: lightweight-charts-python (TradingView charts)
**Alternatives Considered**: PyQtGraph, matplotlib, Plotly
**Rationale**:
- TradingView-quality rendering (industry standard)
- Native PySide6 integration
- Built-in drawing tools (Toolbox)
- Live data streaming support
- Cross-platform compatibility

**Important Consideration**:
Must implement indicators manually - lightweight-charts doesn't include indicator library. Integration with existing `indicators/` module required.

### Decision 4: FastAPI + WebSocket for LEAN Bridge

**Choice**: FastAPI REST API + WebSocket streaming
**Alternatives Considered**: Direct Docker SDK, gRPC, ZeroMQ
**Rationale**:
- FastAPI provides async by default
- WebSocket ideal for real-time quote streaming
- REST for control operations (start backtest, submit orders)
- JSON serialization simplifies debugging
- Potential future web dashboard can reuse same API

**Architecture**:
```
UI ──WebSocket──► FastAPI ──Docker SDK──► LEAN Container
   ◄──Quotes────         ◄──Results────
```

### Decision 5: JSON-Based Workspace Persistence

**Choice**: JSON files for workspace storage
**Alternatives Considered**: SQLite, Qt settings, pickle
**Rationale**:
- Human-readable for debugging
- Easy export/import for backup
- Version control friendly
- Qt's saveState() for dock geometry
- Custom JSON for panel configurations

**Critical Note from Research**:
thinkorswim workspaces are NOT stored on servers - local only. Same pattern recommended here with export capability for backup.

---

## Component Architecture

### High-Level Structure

```
┌────────────────────────────────────────────────────────────────┐
│                     TradingDashboard                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      CDockManager                        │  │
│  │  ┌─────────────┐ ┌─────────────────┐ ┌────────────────┐  │  │
│  │  │  LeftArea   │ │   CentralArea   │ │   RightArea    │  │  │
│  │  │ ┌─────────┐ │ │ ┌─────────────┐ │ │ ┌────────────┐ │  │  │
│  │  │ │Watchlist│ │ │ │ ChartWidget │ │ │ │OptionsChain│ │  │  │
│  │  │ └─────────┘ │ │ │ (lwc-python)│ │ │ └────────────┘ │  │  │
│  │  │ ┌─────────┐ │ │ └─────────────┘ │ │ ┌────────────┐ │  │  │
│  │  │ │ Scanner │ │ │                 │ │ │ OrderEntry │ │  │  │
│  │  │ └─────────┘ │ │                 │ │ └────────────┘ │  │  │
│  │  └─────────────┘ └─────────────────┘ │ ┌────────────┐ │  │  │
│  │                                      │ │ Positions  │ │  │  │
│  │  ┌──────────────────────────────────┐│ └────────────┘ │  │  │
│  │  │           BottomArea             │└────────────────┘  │  │
│  │  │ ┌────────────┐ ┌───────────────┐ │                    │  │
│  │  │ │AgentMonitor│ │  OrderBook    │ │                    │  │
│  │  │ └────────────┘ └───────────────┘ │                    │  │
│  │  └──────────────────────────────────┘                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Data Flow Pattern

```
┌──────────────────────────────────────────────────────────────────┐
│                           UI Layer                               │
│                                                                  │
│  ChartWidget ◄───┐                                               │
│  OptionsChain ◄──┤                                               │
│  Positions ◄─────┤    ┌─────────────────┐                        │
│  AgentMonitor ◄──┼────│ DataController  │                        │
│  OrderBook ◄─────┤    │ (Qt Signals)    │                        │
│                  │    └────────┬────────┘                        │
└──────────────────┼─────────────┼─────────────────────────────────┘
                   │             │
                   │    ┌────────▼────────┐
                   │    │   API Bridge    │
                   │    │  (FastAPI)      │
                   │    └────────┬────────┘
                   │             │
        ┌──────────┴─────────────┼─────────────────┐
        │                        │                 │
┌───────▼───────┐    ┌───────────▼──────┐   ┌──────▼──────┐
│  LEAN Docker  │    │ Decision Logger  │   │  MCP Server │
│   Container   │    │ (Agent Infra)    │   │   (Claude)  │
└───────────────┘    └──────────────────┘   └─────────────┘
```

### Signal/Slot Pattern for Updates

```python
class DataController(QObject):
    """Central data controller using Qt signals."""

    # Market Data Signals
    quote_updated = Signal(str, Quote)       # symbol, quote
    candle_updated = Signal(str, Candle)     # symbol, candle
    chain_updated = Signal(str, OptionChain) # symbol, chain

    # Trading Signals
    order_filled = Signal(Order, Fill)
    position_updated = Signal(str, Position)
    balance_updated = Signal(Balance)

    # Agent Signals
    decision_made = Signal(DecisionLog)
    debate_completed = Signal(DebateResult)
    metrics_updated = Signal(str, AgentMetrics)
```

---

## UI Component Specifications

### 1. Chart Widget (TradingView-Style)

**Requirements**:
- Candlestick, line, area chart types
- Multiple timeframes via toolbar
- Indicator overlays from `indicators/` module
- Drawing tools (trendlines, Fibonacci, rectangles)
- Volume subchart
- Crosshair with price/time display

**Integration with Existing Code**:
```python
# Bridge existing indicators to chart
from indicators.technical_alpha import VWAP, RSI, MACD
from indicators.volatility_bands import BollingerBands, KeltnerChannels

class IndicatorBridge:
    """Bridge existing indicators to lightweight-charts."""

    def apply_vwap(self, chart, data):
        vwap = VWAP()
        values = vwap.calculate(data)
        line = chart.create_line()
        line.set(values)
        line.set_color('#FFD600')
```

### 2. Options Chain Grid

**Requirements**:
- Strike-centered layout (ATM highlighted)
- Calls on left, Puts on right
- Greeks columns: Delta, Gamma, Theta, Vega
- IV, Bid/Ask, Volume, Open Interest
- Expiration selector dropdown
- Click-to-trade from grid
- Strategy builder integration

**Integration with Existing Code**:
```python
from scanners.options_scanner import OptionsScanner

class OptionsChainWidget:
    def __init__(self, scanner: OptionsScanner):
        self.scanner = scanner

    def refresh_chain(self, symbol: str):
        # Use existing scanner for opportunity detection
        opportunities = self.scanner.scan_chain(...)
        self.highlight_opportunities(opportunities)
```

### 3. AI Agent Monitor

**Requirements**:
- Decision timeline (scrolling history)
- Current analysis panel
- Reasoning chain viewer (expandable)
- Bull/Bear debate visualization
- Agent metrics dashboard
- Full audit trail access

**Integration with Existing Code**:
```python
from llm.decision_logger import DecisionLogger
from evaluation.agent_metrics import AgentMetricsTracker
from llm.agents.debate_mechanism import BullBearDebate

class AgentMonitorWidget:
    def __init__(
        self,
        logger: DecisionLogger,
        tracker: AgentMetricsTracker,
        debate: BullBearDebate
    ):
        self.logger = logger
        self.tracker = tracker
        self.debate = debate
```

### 4. Order Entry Panel

**Requirements**:
- One-click buy/sell buttons
- Quantity spinner
- Order type selector (Market, Limit, Stop, Stop-Limit)
- Price input with tick buttons
- TIF selector (DAY, GTC, IOC, FOK)
- Order preview/confirmation

**Integration with Existing Code**:
```python
from execution.smart_execution import SmartExecutor
from models.risk_manager import RiskManager

class OrderEntryWidget:
    def __init__(self, executor: SmartExecutor, risk: RiskManager):
        self.executor = executor
        self.risk = risk

    def submit_order(self, order: Order):
        # Pre-trade validation via risk manager
        if self.risk.validate_order(order):
            self.executor.submit(order)
```

---

## Theme Design (thinkorswim Dark)

### Color Palette

```python
TOS_DARK_THEME = {
    # Base Colors
    "background": "#1E1E1E",
    "panel_bg": "#252525",
    "panel_bg_alt": "#2A2A2A",
    "border": "#3E3E3E",
    "border_light": "#4A4A4A",

    # Text
    "text_primary": "#FFFFFF",
    "text_secondary": "#B0B0B0",
    "text_muted": "#808080",

    # Accent
    "accent": "#00A0FF",
    "accent_hover": "#40B0FF",
    "accent_active": "#0080D0",

    # Trading Colors
    "profit_green": "#00C853",
    "profit_green_bg": "#1B3D2F",
    "loss_red": "#FF1744",
    "loss_red_bg": "#3D1B1B",

    # Warning/Status
    "warning_yellow": "#FFD600",
    "info_blue": "#2196F3",

    # Chart
    "chart_grid": "#2A2A2A",
    "chart_text": "#808080",
    "candle_up": "#00C853",
    "candle_down": "#FF1744",
    "volume_up": "#1B3D2F",
    "volume_down": "#3D1B1B",
}
```

### Qt Stylesheet Structure

```qss
/* Main Window */
QMainWindow {
    background-color: #1E1E1E;
}

/* Dock Widget */
QDockWidget {
    background-color: #252525;
    border: 1px solid #3E3E3E;
    titlebar-close-icon: url(:/icons/close.svg);
}

QDockWidget::title {
    background-color: #2A2A2A;
    padding: 5px;
    font-weight: bold;
}

/* Buttons */
QPushButton {
    background-color: #3E3E3E;
    border: 1px solid #4A4A4A;
    border-radius: 4px;
    padding: 8px 16px;
    color: #FFFFFF;
}

QPushButton:hover {
    background-color: #4A4A4A;
}

/* Buy/Sell Buttons */
QPushButton#buyButton {
    background-color: #00C853;
    border: none;
}

QPushButton#sellButton {
    background-color: #FF1744;
    border: none;
}
```

---

## Implementation Priority Matrix

| Component | Priority | Complexity | Dependencies |
|-----------|----------|------------|--------------|
| Core Framework (PySide6 + Qt-ADS) | P0 | Medium | None |
| Chart Widget | P0 | High | Core Framework |
| LEAN Bridge | P0 | Medium | FastAPI, Docker |
| Theme System | P1 | Low | Core Framework |
| Options Chain | P1 | High | Chart, LEAN Bridge |
| Order Entry | P1 | Medium | LEAN Bridge |
| AI Agent Monitor | P1 | Medium | Existing Agent Infra |
| Position Panel | P1 | Low | LEAN Bridge |
| Workspace Manager | P2 | Medium | All Panels |
| DOM/Price Ladder | P2 | High | LEAN Bridge |
| Strategy Builder | P2 | High | Options Chain |
| thinkScript DSL | P3 | Very High | Chart, Indicators |

---

## Risk Mitigation

### Risk 1: Qt-ADS Learning Curve
**Mitigation**: Start with basic 4-area docking, add Qt-ADS features incrementally

### Risk 2: lightweight-charts Performance with Large Data
**Mitigation**: Implement data decimation, virtual scrolling for historical data

### Risk 3: LEAN Docker Communication Latency
**Mitigation**: Local WebSocket caching, optimistic UI updates

### Risk 4: Cross-Platform Styling Inconsistencies
**Mitigation**: Extensive Qt stylesheet testing on Windows/Linux/macOS

---

## Next Steps

1. **Proof of Concept** (Week 1):
   - Basic PySide6 + Qt-ADS window
   - Single lightweight-chart embedded
   - Mock data streaming

2. **LEAN Integration** (Week 2):
   - FastAPI bridge setup
   - WebSocket quote streaming
   - Backtest trigger from UI

3. **Core Panels** (Week 3-4):
   - Options chain grid
   - Order entry panel
   - Agent monitor (basic)

4. **Polish** (Week 5-6):
   - Theme refinement
   - Workspace save/load
   - Performance optimization

---

## Evidence and Sources

This insight is based on research documented in:
- [TRADING-UI-THINKORSWIM-RESEARCH-DEC2025.md](../research/TRADING-UI-THINKORSWIM-RESEARCH-DEC2025.md)
- [UPGRADE-019-TRADING-UI-THINKORSWIM-STYLE.md](../research/UPGRADE-019-TRADING-UI-THINKORSWIM-STYLE.md)

Key external references:
- [Qt-Advanced-Docking-System](https://github.com/githubuser0xFFFF/Qt-Advanced-Docking-System)
- [lightweight-charts-python](https://github.com/louisnw01/lightweight-charts-python)
- [thinkorswim Flexible Grid](https://toslc.thinkorswim.com/center/howToTos/thinkManual/charts/Flexible-Grid)
- [Datadog LLM Observability](https://www.datadoghq.com/product/llm-observability/)

---

## Changelog

| Date | Change |
|------|--------|
| 2025-12-05 | Initial insight document created |
