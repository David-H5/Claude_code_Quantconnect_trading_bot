"""
Base Options Trading Bot

Shared functionality for all options trading algorithms.
Subclasses implement specific trading strategies.

Common features provided:
- Configuration loading
- Risk management (RiskManager, CircuitBreaker)
- Resource monitoring
- Object store persistence

Subclasses must implement:
- _setup_strategy_specific(): Strategy-specific initialization
- OnData(): Market data processing
"""

from __future__ import annotations

from typing import Any

# QuantConnect imports (available at runtime)
try:
    from AlgorithmImports import *
except ImportError:
    # Stubs for development/testing
    class QCAlgorithm:
        pass

    class Resolution:
        Daily = "Daily"
        Hour = "Hour"
        Minute = "Minute"

    class Slice:
        pass

    class BrokerageName:
        CharlesSchwab = "CharlesSchwab"

    class AccountType:
        Margin = "Margin"

    class OrderEvent:
        pass


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from models import (
    CircuitBreakerConfig,
    RiskLimits,
    RiskManager,
    TradingCircuitBreaker,
)
from observability.monitoring.system.resource import create_resource_monitor
from utils.object_store import create_object_store_manager
from utils.storage_monitor import create_storage_monitor


class BaseOptionsBot(QCAlgorithm):
    """
    Base class for options trading algorithms.

    Provides:
    - Configuration loading
    - Risk management setup
    - Circuit breaker integration
    - Resource monitoring
    - Object store persistence

    Subclasses must implement:
    - _setup_strategy_specific(): Strategy-specific initialization
    - OnData(): Market data processing
    """

    # Override in subclass for different check intervals
    RESOURCE_CHECK_INTERVAL_SECONDS = 30

    def Initialize(self) -> None:
        """Initialize common algorithm components."""
        self._setup_basic()
        self._setup_config()
        self._setup_risk_management()
        self._setup_monitoring()
        self._setup_strategy_specific()  # Subclass hook
        self._setup_common_schedules()
        self._log_initialization()

    def _setup_basic(self) -> None:
        """
        Set up basic algorithm parameters.

        Override in subclass for custom dates/cash:
            def _setup_basic(self) -> None:
                self.SetStartDate(2024, 11, 1)
                self.SetEndDate(2024, 11, 30)
                self.SetCash(100000)
                self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)
        """
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)

        # CRITICAL: Charles Schwab allows ONLY ONE algorithm per account
        # Deploying a second algorithm will automatically stop the first one
        self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

    def _setup_config(self) -> None:
        """Load configuration."""
        try:
            self.config = get_config()
            self.Debug("Configuration loaded successfully")
        except FileNotFoundError:
            self.config = None
            self.Debug("Config file not found, using defaults")

    def _setup_risk_management(self) -> None:
        """Initialize risk management components."""
        risk_config = self._get_config("risk_management", {})

        # Risk limits
        self.risk_limits = RiskLimits(
            max_position_size=risk_config.get("max_position_size_pct", 0.25),
            max_daily_loss=risk_config.get("max_daily_loss_pct", 0.03),
            max_drawdown=risk_config.get("max_drawdown_pct", 0.10),
            max_risk_per_trade=risk_config.get("max_risk_per_trade_pct", 0.02),
        )

        # Risk manager
        self.risk_manager = RiskManager(
            starting_equity=self.Portfolio.TotalPortfolioValue,
            limits=self.risk_limits,
        )

        # Circuit breaker
        breaker_config = CircuitBreakerConfig(
            max_daily_loss_pct=risk_config.get("max_daily_loss_pct", 0.03),
            max_drawdown_pct=risk_config.get("max_drawdown_pct", 0.10),
            max_consecutive_losses=risk_config.get("max_consecutive_losses", 5),
            require_human_reset=risk_config.get("require_human_reset", True),
        )
        self.circuit_breaker = TradingCircuitBreaker(
            config=breaker_config,
            alert_callback=self._on_circuit_breaker_alert,
        )

    def _setup_monitoring(self) -> None:
        """Initialize resource and storage monitoring."""
        # Resource monitor
        resource_config = self._get_config("quantconnect", {}).get("resource_limits", {})
        self.resource_monitor = create_resource_monitor(
            config=resource_config,
            circuit_breaker=self.circuit_breaker,
        )
        self.Debug(f"Resource monitor initialized: {self._get_node_info()}")

        # Tracking timestamp for resource checks
        self._last_resource_check = self.Time

        # Object Store
        object_store_config = self._get_config("quantconnect", {}).get("object_store", {})
        if object_store_config.get("enabled", False):
            self.object_store_manager = create_object_store_manager(
                algorithm=self,
                config=object_store_config,
            )
            self.storage_monitor = create_storage_monitor(
                object_store_manager=self.object_store_manager,
                config=object_store_config,
                circuit_breaker=self.circuit_breaker,
            )
            self.Debug(f"Object Store initialized: {object_store_config.get('tier', 'unknown')} tier")
        else:
            self.object_store_manager = None
            self.storage_monitor = None

    def _setup_strategy_specific(self) -> None:
        """
        Hook for subclass-specific initialization.

        Override in subclass to set up:
        - Executors
        - Scanners
        - LLM integration
        - Custom data subscriptions
        - Schedules
        """
        raise NotImplementedError("Subclass must implement _setup_strategy_specific()")

    def _setup_common_schedules(self) -> None:
        """Set up common scheduled events. Override to add more."""
        # Resource monitoring is handled in OnData to allow flexibility
        pass

    def _log_initialization(self) -> None:
        """Log initialization summary."""
        self.Debug("=" * 60)
        self.Debug(f"{self.__class__.__name__} INITIALIZED")
        self.Debug("=" * 60)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_config(self, section: str, default: Any) -> Any:
        """Get configuration section safely."""
        if self.config:
            return self.config.get(section, default)
        return default

    def _get_node_info(self) -> str:
        """Get current compute node information."""
        qc_config = self._get_config("quantconnect", {})
        nodes = qc_config.get("compute_nodes", {})

        if hasattr(self, "LiveMode") and self.LiveMode:
            node = nodes.get("live_trading", {})
            node_type = "live"
        else:
            node = nodes.get("backtesting", {})
            node_type = "backtest"

        model = node.get("model", "unknown")
        ram_gb = node.get("ram_gb", 0)
        cores = node.get("cores", 0)

        return f"{model} ({cores}C/{ram_gb}GB) [{node_type}]"

    def _check_resources(self) -> None:
        """Check resource usage."""
        if self.resource_monitor:
            try:
                self.resource_monitor.check_resources()
            except Exception as e:
                self.Debug(f"Resource check error: {e}")

    def _should_check_resources(self) -> bool:
        """Determine if it's time to check resources."""
        if (self.Time - self._last_resource_check).total_seconds() >= self.RESOURCE_CHECK_INTERVAL_SECONDS:
            self._last_resource_check = self.Time
            return True
        return False

    def _on_circuit_breaker_alert(self, message: str, details: dict) -> None:
        """
        Handle circuit breaker alerts.

        Args:
            message: Alert message
            details: Dict with keys like 'reason', 'details', 'timestamp'

        Override in subclass for custom handling (email, Discord, etc).
        """
        self.Debug(f"CIRCUIT BREAKER: {message}")

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    def OnEndOfAlgorithm(self) -> None:
        """Common end-of-algorithm reporting."""
        self.Debug("=" * 60)
        self.Debug("ALGORITHM COMPLETED")
        self.Debug(f"Final Equity: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        if hasattr(self, "circuit_breaker"):
            self.Debug(f"Circuit Breaker: {'HALTED' if self.circuit_breaker.is_halted else 'OK'}")
        self.Debug("=" * 60)
