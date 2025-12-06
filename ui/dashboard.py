"""
Trading Dashboard

Main window for the trading bot UI with positions, scanners,
news, charts, and order management.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# LLM Dashboard Widgets (UPGRADE-006 - December 2025)
from .agent_metrics_widget import AgentMetricsWidget, create_agent_metrics_widget
from .debate_viewer import DebateViewer, create_debate_viewer
from .decision_log_viewer import DecisionLogViewer, create_decision_log_viewer
from .evolution_monitor import EvolutionMonitor, create_evolution_monitor
from .widgets import (
    PYSIDE_AVAILABLE,
    AlertData,
    AlertPopup,
    DataTable,
    StatusIndicator,
    StyledButton,
)


if PYSIDE_AVAILABLE:
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import (
        QApplication,
        QDockWidget,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QStatusBar,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )


class PositionsPanel(QWidget if PYSIDE_AVAILABLE else object):
    """Panel displaying current positions."""

    position_selected = Signal(str) if PYSIDE_AVAILABLE else None

    def __init__(self, parent: QWidget | None = None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up positions panel UI."""
        layout = QVBoxLayout(self)

        # Header
        header = QHBoxLayout()
        title = QLabel("<b>Positions</b>")
        title.setStyleSheet("color: white; font-size: 14px;")
        header.addWidget(title)

        refresh_btn = StyledButton("Refresh", "default")
        refresh_btn.clicked.connect(self.refresh)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # Positions table
        self.table = DataTable(["Symbol", "Qty", "Entry", "Current", "P/L", "P/L %", "Actions"])
        layout.addWidget(self.table)

    def refresh(self) -> None:
        """Refresh positions data."""
        # This would fetch from the trading system
        pass

    def update_positions(self, positions: list[dict[str, Any]]) -> None:
        """Update positions display."""
        self.table.clear_data()

        for pos in positions:
            pnl = pos.get("pnl", 0)
            pnl_pct = pos.get("pnl_pct", 0)

            colors = {}
            if pnl > 0:
                colors[4] = "green"
                colors[5] = "green"
            elif pnl < 0:
                colors[4] = "red"
                colors[5] = "red"

            self.table.add_row(
                [
                    pos.get("symbol", ""),
                    pos.get("quantity", 0),
                    f"${pos.get('entry_price', 0):.2f}",
                    f"${pos.get('current_price', 0):.2f}",
                    f"${pnl:,.2f}",
                    f"{pnl_pct:+.2f}%",
                    "Sell",
                ],
                colors,
            )


class ScannerPanel(QWidget if PYSIDE_AVAILABLE else object):
    """Panel for market scanners (options, movement)."""

    def __init__(self, parent: QWidget | None = None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up scanner panel UI."""
        layout = QVBoxLayout(self)

        # Tabs for different scanners
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 8px 16px;
                border: 1px solid #3d3d3d;
            }
            QTabBar::tab:selected {
                background-color: #0066cc;
            }
        """)

        # Options scanner tab
        self.options_table = DataTable(["Symbol", "Type", "Strike", "Expiry", "Bid", "Ask", "IV", "Delta", "Action"])
        tabs.addTab(self.options_table, "Options")

        # Movement scanner tab
        self.movement_table = DataTable(["Symbol", "Change %", "Volume", "News", "Signal", "Action"])
        tabs.addTab(self.movement_table, "Movers")

        layout.addWidget(tabs)

    def update_options(self, options: list[dict[str, Any]]) -> None:
        """Update options scanner display."""
        self.options_table.clear_data()

        for opt in options:
            self.options_table.add_row(
                [
                    opt.get("symbol", ""),
                    opt.get("type", ""),
                    f"${opt.get('strike', 0):.2f}",
                    opt.get("expiry", ""),
                    f"${opt.get('bid', 0):.2f}",
                    f"${opt.get('ask', 0):.2f}",
                    f"{opt.get('iv', 0):.1%}",
                    f"{opt.get('delta', 0):.2f}",
                    "Buy",
                ]
            )

    def update_movers(self, movers: list[dict[str, Any]]) -> None:
        """Update movement scanner display."""
        self.movement_table.clear_data()

        for mov in movers:
            change = mov.get("change_pct", 0)
            colors = {1: "green" if change > 0 else "red"}

            self.movement_table.add_row(
                [
                    mov.get("symbol", ""),
                    f"{change:+.2f}%",
                    f"{mov.get('volume', 0):,}",
                    "Yes" if mov.get("has_news") else "No",
                    mov.get("signal", ""),
                    "Trade",
                ],
                colors,
            )


class NewsPanel(QWidget if PYSIDE_AVAILABLE else object):
    """Panel for news and alerts."""

    def __init__(self, parent: QWidget | None = None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up news panel UI."""
        layout = QVBoxLayout(self)

        # Header
        header = QHBoxLayout()
        title = QLabel("<b>News & Alerts</b>")
        title.setStyleSheet("color: white; font-size: 14px;")
        header.addWidget(title)

        header.addStretch()

        filter_btn = StyledButton("Filter", "default")
        header.addWidget(filter_btn)

        layout.addLayout(header)

        # News table
        self.table = DataTable(["Time", "Symbol", "Headline", "Sentiment", "Action"])
        layout.addWidget(self.table)

    def update_news(self, news_items: list[dict[str, Any]]) -> None:
        """Update news display."""
        self.table.clear_data()

        for item in news_items:
            sentiment = item.get("sentiment", 0)
            colors = {}
            if sentiment > 0.3:
                colors[3] = "green"
            elif sentiment < -0.3:
                colors[3] = "red"

            self.table.add_row(
                [
                    item.get("time", ""),
                    item.get("symbol", ""),
                    item.get("headline", "")[:50] + "...",
                    f"{sentiment:+.2f}",
                    "View",
                ],
                colors,
            )


class OrdersPanel(QWidget if PYSIDE_AVAILABLE else object):
    """Panel for order management."""

    def __init__(self, parent: QWidget | None = None):
        if not PYSIDE_AVAILABLE:
            return

        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up orders panel UI."""
        layout = QVBoxLayout(self)

        # Tabs for order types
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 8px 16px;
            }
            QTabBar::tab:selected {
                background-color: #0066cc;
            }
        """)

        # Open orders
        self.open_orders = DataTable(["Time", "Symbol", "Side", "Qty", "Type", "Price", "Status", "Actions"])
        tabs.addTab(self.open_orders, "Open Orders")

        # Filled orders
        self.filled_orders = DataTable(["Time", "Symbol", "Side", "Qty", "Fill Price", "Status"])
        tabs.addTab(self.filled_orders, "Filled")

        layout.addWidget(tabs)

    def update_open_orders(self, orders: list[dict[str, Any]]) -> None:
        """Update open orders display."""
        self.open_orders.clear_data()

        for order in orders:
            side = order.get("side", "")
            colors = {2: "green" if side == "buy" else "red"}

            self.open_orders.add_row(
                [
                    order.get("time", ""),
                    order.get("symbol", ""),
                    side.upper(),
                    order.get("quantity", 0),
                    order.get("type", ""),
                    f"${order.get('price', 0):.2f}",
                    order.get("status", ""),
                    "Cancel",
                ],
                colors,
            )

    def update_filled_orders(self, orders: list[dict[str, Any]]) -> None:
        """Update filled orders display."""
        self.filled_orders.clear_data()

        for order in orders:
            self.filled_orders.add_row(
                [
                    order.get("time", ""),
                    order.get("symbol", ""),
                    order.get("side", "").upper(),
                    order.get("quantity", 0),
                    f"${order.get('fill_price', 0):.2f}",
                    "Filled",
                ]
            )


class TradingDashboard(QMainWindow if PYSIDE_AVAILABLE else object):
    """
    Main trading dashboard window.

    Features:
    - Dockable panels for positions, scanners, news, orders
    - Real-time price updates
    - Alert popups with action buttons
    - Configurable layout
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        on_sell: Callable | None = None,
        on_buy: Callable | None = None,
    ):
        if not PYSIDE_AVAILABLE:
            print("PySide6 is not installed. Install with: pip install PySide6")
            return

        super().__init__()

        self.config = config or {}
        self.on_sell = on_sell
        self.on_buy = on_buy

        self._alert_popups: list[AlertPopup] = []
        self._setup_ui()
        self._setup_refresh_timer()

    def _setup_ui(self) -> None:
        """Set up the main window UI."""
        self.setWindowTitle("Trading Dashboard")
        self.setMinimumSize(1400, 900)

        # Dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #e0e0e0;
            }
            QDockWidget {
                color: #e0e0e0;
                titlebar-close-icon: url(close.png);
            }
            QDockWidget::title {
                background-color: #2d2d2d;
                padding: 8px;
            }
        """)

        # Menu bar
        self._create_menu_bar()

        # Central widget with main price display
        central = QWidget()
        central_layout = QVBoxLayout(central)

        # Account summary
        summary = QHBoxLayout()

        self.equity_label = QLabel("<b>Equity:</b> $--")
        self.equity_label.setStyleSheet("color: white; font-size: 16px;")
        summary.addWidget(self.equity_label)

        self.pnl_label = QLabel("<b>Day P/L:</b> $--")
        self.pnl_label.setStyleSheet("color: white; font-size: 16px;")
        summary.addWidget(self.pnl_label)

        self.status = StatusIndicator("System", "active")
        summary.addWidget(self.status)

        summary.addStretch()

        central_layout.addLayout(summary)
        central_layout.addStretch()

        self.setCentralWidget(central)

        # Create dock widgets
        self._create_docks()

        # Status bar
        status_bar = QStatusBar()
        status_bar.setStyleSheet("background-color: #2d2d2d; color: #e0e0e0;")
        status_bar.showMessage("Ready")
        self.setStatusBar(status_bar)

    def _create_menu_bar(self) -> None:
        """Create menu bar."""
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet("""
            QMenuBar {
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
            QMenuBar::item:selected {
                background-color: #0066cc;
            }
            QMenu {
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
            QMenu::item:selected {
                background-color: #0066cc;
            }
        """)

        # File menu
        file_menu = menu_bar.addMenu("File")

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self._show_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menu_bar.addMenu("View")

        # LLM Dashboard submenu (UPGRADE-006)
        llm_menu = view_menu.addMenu("LLM Dashboard")

        self._llm_dock_actions: dict[str, QAction] = {}
        for dock_name in ["Agent Metrics", "Debate Viewer", "Evolution Monitor", "Decision Log"]:
            action = QAction(dock_name, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, name=dock_name: self._toggle_llm_dock(name, checked))
            llm_menu.addAction(action)
            self._llm_dock_actions[dock_name] = action

        llm_menu.addSeparator()
        show_all_action = QAction("Show All LLM Panels", self)
        show_all_action.triggered.connect(self._show_all_llm_docks)
        llm_menu.addAction(show_all_action)

        hide_all_action = QAction("Hide All LLM Panels", self)
        hide_all_action.triggered.connect(self._hide_all_llm_docks)
        llm_menu.addAction(hide_all_action)

        view_menu.addSeparator()

        # Trading menu
        trading_menu = menu_bar.addMenu("Trading")

        halt_action = QAction("Halt Trading", self)
        halt_action.triggered.connect(self._halt_trading)
        trading_menu.addAction(halt_action)

        resume_action = QAction("Resume Trading", self)
        resume_action.triggered.connect(self._resume_trading)
        trading_menu.addAction(resume_action)

    def _create_docks(self) -> None:
        """Create dockable panels."""
        # Positions dock
        positions_dock = QDockWidget("Positions", self)
        self.positions_panel = PositionsPanel()
        positions_dock.setWidget(self.positions_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, positions_dock)

        # Scanner dock
        scanner_dock = QDockWidget("Scanners", self)
        self.scanner_panel = ScannerPanel()
        scanner_dock.setWidget(self.scanner_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, scanner_dock)

        # News dock
        news_dock = QDockWidget("News", self)
        self.news_panel = NewsPanel()
        news_dock.setWidget(self.news_panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, news_dock)

        # Orders dock
        orders_dock = QDockWidget("Orders", self)
        self.orders_panel = OrdersPanel()
        orders_dock.setWidget(self.orders_panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, orders_dock)

        # Tab the bottom docks
        self.tabifyDockWidget(news_dock, orders_dock)

        # LLM Dashboard docks (UPGRADE-006)
        self._create_llm_docks()

    def _create_llm_docks(self) -> None:
        """Create LLM dashboard docks (UPGRADE-006)."""
        # Agent Metrics dock
        metrics_dock = QDockWidget("Agent Metrics", self)
        self.agent_metrics_widget = AgentMetricsWidget()
        metrics_dock.setWidget(self.agent_metrics_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, metrics_dock)
        metrics_dock.hide()  # Hidden by default, shown via View menu

        # Debate Viewer dock
        debate_dock = QDockWidget("Debate Viewer", self)
        self.debate_viewer = DebateViewer()
        debate_dock.setWidget(self.debate_viewer)
        self.addDockWidget(Qt.RightDockWidgetArea, debate_dock)
        debate_dock.hide()

        # Evolution Monitor dock
        evolution_dock = QDockWidget("Evolution Monitor", self)
        self.evolution_monitor = EvolutionMonitor()
        evolution_dock.setWidget(self.evolution_monitor)
        self.addDockWidget(Qt.RightDockWidgetArea, evolution_dock)
        evolution_dock.hide()

        # Decision Log dock
        decision_dock = QDockWidget("Decision Log", self)
        self.decision_log_viewer = DecisionLogViewer()
        decision_dock.setWidget(self.decision_log_viewer)
        self.addDockWidget(Qt.BottomDockWidgetArea, decision_dock)
        decision_dock.hide()

        # Store docks for View menu
        self._llm_docks = {
            "Agent Metrics": metrics_dock,
            "Debate Viewer": debate_dock,
            "Evolution Monitor": evolution_dock,
            "Decision Log": decision_dock,
        }

    def _setup_refresh_timer(self) -> None:
        """Set up auto-refresh timer."""
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_data)

        refresh_rate = self.config.get("ui", {}).get("refresh_rate_ms", 1000)
        self.refresh_timer.start(refresh_rate)

    def _refresh_data(self) -> None:
        """Refresh all data displays."""
        # This would be connected to data sources
        pass

    def _show_settings(self) -> None:
        """Show settings dialog."""
        # Settings dialog implementation
        QMessageBox.information(self, "Settings", "Settings dialog coming soon")

    def _halt_trading(self) -> None:
        """Halt all trading."""
        result = QMessageBox.question(
            self,
            "Halt Trading",
            "Are you sure you want to halt all trading?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if result == QMessageBox.Yes:
            self.status.set_status("error")
            self.statusBar().showMessage("Trading halted")

    def _resume_trading(self) -> None:
        """Resume trading."""
        self.status.set_status("active")
        self.statusBar().showMessage("Trading resumed")

    def _toggle_llm_dock(self, dock_name: str, show: bool) -> None:
        """Toggle LLM dock visibility (UPGRADE-006)."""
        if hasattr(self, "_llm_docks") and dock_name in self._llm_docks:
            dock = self._llm_docks[dock_name]
            if show:
                dock.show()
            else:
                dock.hide()

    def _show_all_llm_docks(self) -> None:
        """Show all LLM dashboard docks (UPGRADE-006)."""
        if hasattr(self, "_llm_docks"):
            for dock in self._llm_docks.values():
                dock.show()
        if hasattr(self, "_llm_dock_actions"):
            for action in self._llm_dock_actions.values():
                action.setChecked(True)

    def _hide_all_llm_docks(self) -> None:
        """Hide all LLM dashboard docks (UPGRADE-006)."""
        if hasattr(self, "_llm_docks"):
            for dock in self._llm_docks.values():
                dock.hide()
        if hasattr(self, "_llm_dock_actions"):
            for action in self._llm_dock_actions.values():
                action.setChecked(False)

    def set_llm_components(
        self,
        metrics_tracker: Any | None = None,
        debate_mechanism: Any | None = None,
        evolving_agents: list[Any] | None = None,
        decision_logger: Any | None = None,
    ) -> None:
        """
        Configure LLM dashboard components (UPGRADE-006).

        Args:
            metrics_tracker: AgentMetricsTracker instance
            debate_mechanism: BullBearDebate instance
            evolving_agents: List of SelfEvolvingAgent instances
            decision_logger: DecisionLogger instance
        """
        if metrics_tracker and hasattr(self, "agent_metrics_widget"):
            self.agent_metrics_widget.set_metrics_tracker(metrics_tracker)

        if debate_mechanism and hasattr(self, "debate_viewer"):
            self.debate_viewer.set_debate_mechanism(debate_mechanism)

        if evolving_agents and hasattr(self, "evolution_monitor"):
            self.evolution_monitor.set_evolving_agents(evolving_agents)

        if decision_logger and hasattr(self, "decision_log_viewer"):
            self.decision_log_viewer.set_decision_logger(decision_logger)

    def update_equity(self, equity: float, day_pnl: float) -> None:
        """Update account equity display."""
        self.equity_label.setText(f"<b>Equity:</b> ${equity:,.2f}")

        color = "#28a745" if day_pnl >= 0 else "#dc3545"
        sign = "+" if day_pnl >= 0 else ""
        self.pnl_label.setText(f"<b>Day P/L:</b> <span style='color:{color}'>{sign}${day_pnl:,.2f}</span>")

    def show_alert(self, alert: AlertData) -> None:
        """Show an alert popup."""
        if not self.config.get("ui", {}).get("popup_alerts", True):
            return

        popup = AlertPopup(alert, self)
        popup.closed.connect(lambda: self._alert_popups.remove(popup))
        self._alert_popups.append(popup)

        # Position popup
        popup.move(self.width() - popup.width() - 20, 80 + len(self._alert_popups) * 150)
        popup.show()


def create_dashboard(
    config: dict[str, Any] | None = None,
    on_sell: Callable | None = None,
    on_buy: Callable | None = None,
) -> TradingDashboard | None:
    """
    Create and return a trading dashboard instance.

    Args:
        config: Dashboard configuration
        on_sell: Callback for sell actions
        on_buy: Callback for buy actions

    Returns:
        TradingDashboard instance or None if PySide6 unavailable
    """
    if not PYSIDE_AVAILABLE:
        print("PySide6 is required for the dashboard. Install with: pip install PySide6")
        return None

    return TradingDashboard(config, on_sell, on_buy)


def run_dashboard(dashboard: TradingDashboard) -> int:
    """
    Run the dashboard application.

    Args:
        dashboard: Dashboard instance to run

    Returns:
        Exit code
    """
    if not PYSIDE_AVAILABLE:
        return 1

    import sys

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dashboard.show()
    return app.exec()


def create_llm_dashboard(
    metrics_tracker: Any | None = None,
    debate_mechanism: Any | None = None,
    evolving_agents: list[Any] | None = None,
    decision_logger: Any | None = None,
    config: dict[str, Any] | None = None,
) -> TradingDashboard | None:
    """
    Create a trading dashboard with LLM components pre-configured (UPGRADE-006).

    Args:
        metrics_tracker: AgentMetricsTracker instance
        debate_mechanism: BullBearDebate instance
        evolving_agents: List of SelfEvolvingAgent instances
        decision_logger: DecisionLogger instance
        config: Dashboard configuration

    Returns:
        Configured TradingDashboard with LLM panels visible

    Example:
        from evaluation.agent_metrics import AgentMetricsTracker
        from llm.agents.debate_mechanism import BullBearDebate
        from llm.decision_logger import DecisionLogger

        tracker = AgentMetricsTracker()
        debate = BullBearDebate()
        logger = DecisionLogger()

        dashboard = create_llm_dashboard(
            metrics_tracker=tracker,
            debate_mechanism=debate,
            decision_logger=logger,
        )
        run_dashboard(dashboard)
    """
    if not PYSIDE_AVAILABLE:
        print("PySide6 is required for the dashboard. Install with: pip install PySide6")
        return None

    dashboard = TradingDashboard(config=config)

    # Configure LLM components
    dashboard.set_llm_components(
        metrics_tracker=metrics_tracker,
        debate_mechanism=debate_mechanism,
        evolving_agents=evolving_agents,
        decision_logger=decision_logger,
    )

    # Show LLM panels
    dashboard._show_all_llm_docks()

    return dashboard


__all__ = [
    "PositionsPanel",
    "ScannerPanel",
    "NewsPanel",
    "OrdersPanel",
    "TradingDashboard",
    "create_dashboard",
    "run_dashboard",
    # LLM Dashboard (UPGRADE-006)
    "AgentMetricsWidget",
    "DebateViewer",
    "EvolutionMonitor",
    "DecisionLogViewer",
    "create_llm_dashboard",
    "create_agent_metrics_widget",
    "create_debate_viewer",
    "create_evolution_monitor",
    "create_decision_log_viewer",
]
