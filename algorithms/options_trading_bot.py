"""
Options Trading Bot - Main QuantConnect Algorithm

A comprehensive options trading bot with:
- Charles Schwab brokerage integration
- LLM-powered news analysis
- Underpriced options scanner
- Movement scanner with news corroboration
- Profit-taking risk management
- Smart order execution with cancel/replace

Uses the QuantConnect Algorithm Framework architecture:
Universe → Alpha → Portfolio → Risk → Execution
"""

from datetime import timedelta
from typing import Any


# QuantConnect imports (available at runtime)
try:
    from AlgorithmImports import *
except ImportError:
    # Stubs for development
    class QCAlgorithm:
        pass

    class Resolution:
        Daily = "Daily"
        Hour = "Hour"
        Minute = "Minute"

    class UniverseSettings:
        pass

    class InsightDirection:
        Up = 1
        Down = -1
        Flat = 0

    class InsightType:
        Price = "Price"


# Local imports
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from execution import (
    create_profit_taking_model,
    create_smart_execution_model,
)
from indicators import TechnicalAlphaModel as TechAlpha
from models import (
    CircuitBreakerConfig,
    RiskLimits,
    RiskManager,
    TradingCircuitBreaker,
)
from utils.object_store import (
    StorageCategory,
    create_object_store_manager,
)
from utils.resource_monitor import create_resource_monitor
from utils.storage_monitor import create_storage_monitor


class OptionsTradingBot(QCAlgorithm):
    """
    Main trading algorithm with full framework integration.

    Architecture:
    1. Universe Selection - Top liquid stocks + options
    2. Alpha Generation - Technical + LLM sentiment signals
    3. Portfolio Construction - Options-focused allocation
    4. Risk Management - Circuit breaker + profit taking
    5. Execution - Smart cancel/replace
    """

    def Initialize(self) -> None:
        """Initialize algorithm parameters and framework components."""
        # Basic setup
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)

        # Brokerage selection
        # CRITICAL: Charles Schwab allows ONLY ONE algorithm per account
        # Deploying a second algorithm will automatically stop the first one
        # All strategies must be combined into this single algorithm
        self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

        # Load configuration
        try:
            self.config = get_config()
        except FileNotFoundError:
            self.config = None
            self.Debug("Config file not found, using defaults")

        # Initialize risk limits
        risk_config = self._get_config("risk_management", {})
        self.risk_limits = RiskLimits(
            max_position_size=risk_config.get("max_position_size_pct", 0.25),
            max_daily_loss=risk_config.get("max_daily_loss_pct", 0.03),
            max_drawdown=risk_config.get("max_drawdown_pct", 0.10),
            max_risk_per_trade=risk_config.get("max_risk_per_trade_pct", 0.02),
        )

        # Initialize risk manager
        self.risk_manager = RiskManager(
            starting_equity=self.Portfolio.TotalPortfolioValue,
            limits=self.risk_limits,
        )

        # Initialize circuit breaker
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

        # Initialize resource monitor
        resource_config = self._get_config("quantconnect", {}).get("resource_limits", {})
        self.resource_monitor = create_resource_monitor(
            config=resource_config,
            circuit_breaker=self.circuit_breaker,
        )
        self.Debug(f"Resource monitor initialized for node: {self._get_node_info()}")

        # Initialize Object Store manager
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

            # Load any saved trading state
            self._restore_trading_state()
        else:
            self.object_store_manager = None
            self.storage_monitor = None

        # Initialize profit-taking model
        profit_config = self._get_config("profit_taking", {})
        if profit_config.get("enabled", True):
            self.profit_taker = create_profit_taking_model(
                config=self.config.get_profit_taking_config() if self.config else None,
                order_callback=self._on_profit_take_order,
            )
        else:
            self.profit_taker = None

        # Initialize smart execution
        exec_config = self._get_config("order_execution", {})
        if exec_config.get("cancel_replace_enabled", True):
            self.smart_executor = create_smart_execution_model(
                config=self.config.get_order_execution_config() if self.config else None,
                order_callback=self._on_order_event,
            )
        else:
            self.smart_executor = None

        # Technical analysis
        self.technical = {}  # Symbol -> TechAlpha

        # Tracking
        self._pending_orders = {}
        self._consecutive_losses = 0
        self._daily_pnl = 0.0
        self._last_trade_date = None

        # Universe
        self._setup_universe()

        # Schedule functions
        self._setup_schedules()

        # Warm-up period for indicators
        # Note: As of LEAN PR #6720, Greeks calculations use IV and require NO warmup
        # This warmup is for technical indicators (RSI, MACD, etc.) only
        self.SetWarmUp(timedelta(days=50))

        # Initial resource check
        self._check_resources()

        self.Debug("Options Trading Bot initialized")

    def _get_config(self, section: str, default: Any) -> Any:
        """Get configuration section safely."""
        if self.config:
            return self.config.get(section, default)
        return default

    def _get_node_info(self) -> str:
        """Get current node information from config."""
        qc_config = self._get_config("quantconnect", {})
        nodes = qc_config.get("compute_nodes", {})

        # Determine which node we're running on based on context
        # In live trading, use live_trading node; otherwise use backtesting
        if hasattr(self, "LiveMode") and self.LiveMode:
            node = nodes.get("live_trading", {})
            node_type = "live_trading"
        else:
            node = nodes.get("backtesting", {})
            node_type = "backtesting"

        model = node.get("model", "unknown")
        ram_gb = node.get("ram_gb", 0)
        cores = node.get("cores", 0)

        return f"{model} ({cores} cores, {ram_gb}GB RAM) [{node_type}]"

    def _setup_universe(self) -> None:
        """Set up the trading universe."""
        # Add SPY as baseline
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol

        # Add watchlist equities (configurable)
        watchlist = self._get_config("news_alerts", {}).get("watchlist", [])
        for symbol in watchlist[:20]:  # Limit for performance
            try:
                equity = self.AddEquity(symbol, Resolution.Minute)
                self.technical[equity.Symbol] = TechAlpha(self._get_config("technical_indicators", {}))
            except Exception as e:
                self.Debug(f"Failed to add {symbol}: {e}")

        # Option chain universe
        self.option_symbols = {}
        scanner_config = self._get_config("options_scanner", {})
        if scanner_config.get("enabled", True):
            # Add options for main symbols
            for symbol in list(self.technical.keys())[:5]:
                try:
                    option = self.AddOption(symbol, Resolution.Minute)
                    option.SetFilter(self._option_filter)
                    self.option_symbols[symbol] = option.Symbol
                except (ValueError, KeyError) as e:
                    self.Debug(f"Failed to add option for {symbol}: {e}")
                    continue

    def _option_filter(self, universe: OptionFilterUniverse) -> OptionFilterUniverse:
        """
        Filter option contracts using Greeks and IV before detailed analysis.

        This pre-filters the option universe to reduce data processing in scanners.
        Greeks are available immediately (IV-based, no warmup required per LEAN PR #6720).

        Note: For multi-leg strategies (butterflies, condors, spreads), use ComboOrders
        for atomic execution or OptionStrategies factory methods.

        Example ComboOrder usage:
            legs = [
                Leg.Create(call1_symbol, 1),   # Buy call
                Leg.Create(call2_symbol, -2),  # Sell 2 calls
                Leg.Create(call3_symbol, 1),   # Buy call (butterfly)
            ]
            self.ComboLimitOrder(legs, quantity=1, limit_price=net_debit)

        Example OptionStrategies usage:
            strategy = OptionStrategies.butterfly_call(
                self.option_symbol, lower_strike, middle_strike, upper_strike, expiry
            )
            self.buy(strategy, 1)  # Atomic execution with automatic position grouping

        Available ComboOrder types: ComboMarket, ComboLimit (Schwab supports these)
        NOT supported on Schwab: ComboLegLimitOrder (individual leg limits)
        """
        scanner_config = self._get_config("options_scanner", {})
        min_dte = scanner_config.get("min_days_to_expiry", 7)
        max_dte = scanner_config.get("max_days_to_expiry", 45)
        min_delta = scanner_config.get("min_delta", 0.25)
        max_delta = scanner_config.get("max_delta", 0.35)
        min_iv = scanner_config.get("min_implied_volatility", 0.20)

        # Chainable Greeks-based filtering (reduces data before scanner processing)
        return (
            universe.IncludeWeeklys()
            .Strikes(-10, 10)  # Strike range from ATM
            .Expiration(min_dte, max_dte)  # DTE range
            .Delta(min_delta, max_delta)  # Delta range for scanner
            .ImpliedVolatility(min_iv, None)
        )  # Minimum IV threshold

    def _setup_schedules(self) -> None:
        """Set up scheduled events."""
        # Reset daily counters
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(9, 30),
            self._reset_daily_counters,
        )

        # End of day risk check
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.BeforeMarketClose("SPY", 15),
            self._end_of_day_check,
        )

        # Process unfilled orders every 30 seconds
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(seconds=30)),
            self._process_unfilled_orders,
        )

        # Monitor resources every 30 seconds
        monitor_config = self._get_config("quantconnect", {}).get("monitoring", {})
        if monitor_config.get("enabled", True):
            interval_seconds = monitor_config.get("check_interval_seconds", 30)
            self.Schedule.On(
                self.DateRules.EveryDay(),
                self.TimeRules.Every(timedelta(seconds=interval_seconds)),
                self._check_resources,
            )

        # Monitor storage every 6 hours
        if self.storage_monitor:
            storage_monitor_config = (
                self._get_config("quantconnect", {}).get("object_store", {}).get("usage_monitoring", {})
            )
            check_interval_hours = storage_monitor_config.get("check_interval_hours", 6)
            self.Schedule.On(
                self.DateRules.EveryDay(),
                self.TimeRules.Every(timedelta(hours=check_interval_hours)),
                self._check_storage,
            )

        # Save trading state daily at end of day
        if self.object_store_manager:
            self.Schedule.On(
                self.DateRules.EveryDay(),
                self.TimeRules.BeforeMarketClose("SPY", 5),
                self._save_trading_state,
            )

    def _reset_daily_counters(self) -> None:
        """Reset daily tracking counters."""
        self._daily_pnl = 0.0
        self._last_trade_date = self.Time.date()

    def _end_of_day_check(self) -> None:
        """Perform end of day risk checks."""
        self.risk_manager.update_equity(self.Portfolio.TotalPortfolioValue)

    def _process_unfilled_orders(self) -> None:
        """Process unfilled orders for cancel/replace."""
        if not self.smart_executor:
            return

        # Get quotes for pending orders
        quote_data = {}
        for order_id, order in self._pending_orders.items():
            symbol = order.symbol
            if symbol in self.Securities:
                security = self.Securities[symbol]
                quote_data[symbol] = (security.BidPrice, security.AskPrice)

        # Process with smart executor
        replaced = self.smart_executor.process_unfilled_orders(quote_data)
        for order in replaced:
            self.Debug(f"Order replaced: {order.order_id}")

    def _check_resources(self) -> None:
        """Monitor system resources and alert if thresholds exceeded."""
        if not hasattr(self, "resource_monitor"):
            return

        # Count active securities and positions
        active_securities = len(self.Securities)
        active_positions = sum(1 for h in self.Portfolio.Values if h.Invested)

        # Update resource monitor
        metrics = self.resource_monitor.update(
            active_securities=active_securities,
            active_positions=active_positions,
        )

        # Check if healthy
        if not self.resource_monitor.is_healthy():
            self.Debug("RESOURCE WARNING: System resources under pressure")
            self.Debug(f"  Memory: {metrics.memory_pct:.1f}%")
            self.Debug(f"  CPU: {metrics.cpu_pct:.1f}%")
            self.Debug(f"  Securities: {active_securities}")

            # Get recent alerts
            alerts = self.resource_monitor.get_recent_alerts(limit=3)
            for alert in alerts:
                self.Debug(f"  Alert: {alert.message}")

    def _check_storage(self) -> None:
        """Monitor Object Store usage and cleanup if needed."""
        if not self.storage_monitor:
            return

        stats = self.storage_monitor.check_usage()

        self.Debug(f"Object Store: {stats['total_gb']:.2f}/{stats['limit_gb']}GB ({stats['total_pct']:.1f}%)")
        self.Debug(f"Files: {stats['file_count']:,}/{stats['file_limit']:,}")

        # Check if cleanup needed
        if not self.storage_monitor.is_healthy():
            self.Debug("STORAGE WARNING: Performing cleanup")
            suggestions = self.storage_monitor.suggest_cleanup()

            # Execute cleanup
            self.object_store_manager.cleanup_expired()

            # Get alerts
            alerts = self.storage_monitor.get_recent_alerts(limit=3)
            for alert in alerts:
                self.Debug(f"  Storage Alert: {alert.message}")

    def _save_trading_state(self) -> None:
        """Save current trading state to Object Store."""
        if not self.object_store_manager:
            return

        try:
            # Prepare state snapshot
            state = {
                "timestamp": str(self.Time),
                "equity": self.Portfolio.TotalPortfolioValue,
                "daily_pnl": self._daily_pnl,
                "positions": {
                    str(holding.Symbol): {
                        "quantity": holding.Quantity,
                        "avg_price": holding.AveragePrice,
                        "unrealized_pnl": holding.UnrealizedProfit,
                    }
                    for holding in self.Portfolio.Values
                    if holding.Invested
                },
                "circuit_breaker": self.circuit_breaker.get_status(),
                "resource_stats": self.resource_monitor.get_statistics() if self.resource_monitor else {},
            }

            # Save with 90-day expiration
            self.object_store_manager.save(
                key=f"trading_state_{self.Time.date()}",
                data=state,
                category=StorageCategory.TRADING_STATE,
                expire_days=90,
            )

            self.Debug(f"Trading state saved for {self.Time.date()}")

        except Exception as e:
            self.Error(f"Failed to save trading state: {e}")

    def _restore_trading_state(self) -> None:
        """Restore trading state from Object Store on initialization."""
        if not self.object_store_manager:
            return

        try:
            # Find most recent state
            keys = self.object_store_manager.list_keys(StorageCategory.TRADING_STATE)
            if not keys:
                self.Debug("No previous trading state found")
                return

            # Sort by date and get most recent
            state_keys = [k for k in keys if k.startswith("trading_state_")]
            if not state_keys:
                return

            latest_key = sorted(state_keys)[-1]
            state = self.object_store_manager.load(latest_key)

            if state:
                self.Debug(f"Restored trading state from {latest_key}")
                # Can use state data to restore circuit breaker, etc.
                # For safety, we don't auto-restore positions

        except Exception as e:
            self.Error(f"Failed to restore trading state: {e}")

    def OnData(self, data: Slice) -> None:
        """Process incoming market data."""
        if self.IsWarmingUp:
            return

        # Check circuit breaker
        if not self._check_circuit_breaker():
            return

        # Update technical indicators
        for symbol, tech in self.technical.items():
            if symbol in data.Bars:
                bar = data.Bars[symbol]
                tech.update(
                    high=bar.High,
                    low=bar.Low,
                    close=bar.Close,
                    volume=bar.Volume,
                    timestamp=self.Time,
                )

        # Generate equity signals
        signals = self._generate_signals(data)

        # Execute trades based on signals
        for signal in signals:
            self._execute_signal(signal, data)

        # Process option chains for opportunities
        self._process_options_chains(data)

        # Check profit-taking on existing positions
        self._check_profit_taking(data)

    def _check_circuit_breaker(self) -> bool:
        """Check if trading is allowed by circuit breaker."""
        portfolio = {
            "daily_pnl_pct": self._daily_pnl / self.Portfolio.TotalPortfolioValue
            if self.Portfolio.TotalPortfolioValue > 0
            else 0,
            "current_equity": self.Portfolio.TotalPortfolioValue,
            "peak_equity": self.risk_manager.peak_equity,
        }

        can_trade, reason = self.circuit_breaker.check(portfolio)

        if not can_trade:
            self.Debug(f"Circuit breaker tripped: {reason}")
            return False

        return True

    def _generate_signals(self, data: Slice) -> list[dict[str, Any]]:
        """Generate trading signals from technical and LLM analysis."""
        signals = []

        for symbol, tech in self.technical.items():
            if symbol not in data.Bars:
                continue

            bar = data.Bars[symbol]

            # Get technical signals
            tech_signals = tech.generate_signals(bar.Close)
            composite = tech.get_composite_signal(tech_signals)

            # Only trade on strong signals
            if composite.strength < 0.6:
                continue

            signal = {
                "symbol": symbol,
                "direction": 1 if composite.signal.value > 0 else -1,
                "strength": composite.strength,
                "price": bar.Close,
                "type": "equity",  # or "option"
            }
            signals.append(signal)

        return signals

    def _process_options_chains(self, data: Slice) -> None:
        """
        Process option chains for trading opportunities using Greeks.

        Greeks are available immediately (IV-based, no warmup required per PR #6720).
        Use theta_per_day instead of theta for Interactive Brokers compatibility.
        """
        scanner_config = self._get_config("options_scanner", {})
        if not scanner_config.get("enabled", True):
            return

        # Process each option chain
        for chain in data.OptionChains.Values:
            if len(chain.Contracts) == 0:
                continue

            # Get underlying price for reference
            underlying_price = chain.Underlying.Price

            # Scan for underpriced options using Greeks
            for contract in chain:
                # Greeks available immediately (no warmup needed)
                delta = contract.Greeks.Delta
                gamma = contract.Greeks.Gamma
                theta_per_day = contract.Greeks.ThetaPerDay  # Use for IB compatibility
                vega = contract.Greeks.Vega
                iv = contract.ImpliedVolatility

                # Skip if no bid/ask data
                if contract.BidPrice == 0 or contract.AskPrice == 0:
                    continue

                # Example: Identify underpriced high IV options
                # (This is a simple example - use your scanners for detailed analysis)
                mid_price = (contract.BidPrice + contract.AskPrice) / 2

                # Check for underpriced opportunities
                if iv > scanner_config.get("min_implied_volatility", 0.20):
                    # Calculate simple underpricing metric
                    # (In production, use your options_scanner module)
                    intrinsic_value = max(
                        0,
                        (underlying_price - contract.Strike)
                        if contract.Right == OptionRight.Call
                        else (contract.Strike - underlying_price),
                    )
                    time_value = mid_price - intrinsic_value

                    # Example signal: High IV with reasonable time value
                    if time_value > 0 and abs(delta) > 0.25:
                        self.Debug(
                            f"Option opportunity: {contract.Symbol.Value} "
                            f"Delta={delta:.3f} IV={iv:.1%} ThetaPerDay={theta_per_day:.4f}"
                        )

                        # NOTE: For actual trading, integrate with your scanners:
                        # from scanners import create_options_scanner
                        # opportunities = self.options_scanner.scan_chain(...)

            # Example: Multi-leg strategy using OptionStrategies factory methods
            # (Uncomment and customize for your strategy)
            """
            # Get ATM calls for butterfly
            expiry = min([c.Expiry for c in chain])
            calls = sorted([c for c in chain
                if c.Expiry == expiry and c.Right == OptionRight.Call],
                key=lambda x: x.Strike)

            if len(calls) >= 5:
                # Create butterfly using factory method
                strategy = OptionStrategies.butterfly_call(
                    chain.Symbol,
                    calls[0].Strike,   # Lower strike
                    calls[2].Strike,   # Middle strike (ATM)
                    calls[4].Strike,   # Upper strike
                    expiry
                )

                # Execute atomically with automatic position grouping
                # self.buy(strategy, 1)

                # OR use ComboLimitOrder for price control:
                legs = [
                    Leg.Create(calls[0].Symbol, 1),
                    Leg.Create(calls[2].Symbol, -2),
                    Leg.Create(calls[4].Symbol, 1),
                ]
                # self.ComboLimitOrder(legs, quantity=1, limit_price=net_debit)
            """

    def _execute_signal(self, signal: dict[str, Any], data: Slice) -> None:
        """Execute a trading signal."""
        symbol = signal["symbol"]
        direction = signal["direction"]
        strength = signal["strength"]
        price = signal["price"]

        # Check position limits
        if not self.risk_manager.can_take_position(symbol, price * 100):
            return

        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            price=price,
            stop_loss_pct=0.05,
        )

        if position_size <= 0:
            return

        # Determine order side
        quantity = int(position_size * direction)

        if quantity == 0:
            return

        # Place order
        if self.smart_executor:
            # Use smart execution
            security = self.Securities[symbol]
            order = self.smart_executor.submit_order(
                symbol=str(symbol),
                side="buy" if quantity > 0 else "sell",
                quantity=abs(quantity),
                bid_price=security.BidPrice,
                ask_price=security.AskPrice,
            )
            self._pending_orders[order.order_id] = order
        else:
            # Standard market order
            self.MarketOrder(symbol, quantity)

        # Register with profit taker
        if self.profit_taker and quantity > 0:
            self.profit_taker.register_position(str(symbol), price, abs(quantity))

    def _check_profit_taking(self, data: Slice) -> None:
        """Check profit-taking thresholds for existing positions."""
        if not self.profit_taker:
            return

        for holding in self.Portfolio.Values:
            if not holding.Invested:
                continue

            symbol = holding.Symbol
            if symbol not in data.Bars:
                continue

            current_price = data.Bars[symbol].Close
            quantity = holding.Quantity

            # Update and get profit-take orders
            orders = self.profit_taker.update_position(str(symbol), current_price, quantity)

            for order in orders:
                self.MarketOrder(symbol, -order.quantity)
                self.Debug(f"Profit-take: {order.symbol} x{order.quantity} at {order.current_gain_pct:.1%}")

    def OnOrderEvent(self, orderEvent: OrderEvent) -> None:
        """Handle order events."""
        if orderEvent.Status == OrderStatus.Filled:
            symbol = orderEvent.Symbol
            fill_price = orderEvent.FillPrice
            quantity = orderEvent.FillQuantity

            # Update risk manager
            self.risk_manager.update_position(
                str(symbol),
                quantity,
                fill_price,
            )

            # Track P/L
            if symbol in self.Portfolio:
                holding = self.Portfolio[symbol]
                if holding.Quantity == 0:
                    # Position closed
                    pnl = orderEvent.FillQuantity * (fill_price - holding.AveragePrice)
                    self._daily_pnl += pnl

                    # Update circuit breaker
                    is_winner = pnl > 0
                    self.circuit_breaker.record_trade_result(is_winner)

            # Update smart executor
            if self.smart_executor:
                for order_id, order in list(self._pending_orders.items()):
                    if order.symbol == str(symbol):
                        self.smart_executor.update_order_status(
                            order_id,
                            "filled",
                            orderEvent.FillQuantity,
                            fill_price,
                        )
                        del self._pending_orders[order_id]

    def _on_circuit_breaker_alert(self, message: str, details: dict) -> None:
        """Handle circuit breaker alerts."""
        self.Debug(f"CIRCUIT BREAKER: {message}")
        self.Log(f"Circuit breaker details: {details}")

        # Liquidate all positions
        self.Liquidate()

    def _on_profit_take_order(self, order: Any) -> None:
        """Handle profit-take order callback."""
        self.Debug(f"Profit-take triggered: {order.symbol} at {order.current_gain_pct:.1%}")

    def _on_order_event(self, order: Any, action: str) -> None:
        """Handle smart execution order events."""
        self.Debug(f"Order {action}: {order.order_id} for {order.symbol}")

    def OnEndOfDay(self, symbol: Symbol) -> None:
        """End of day processing."""
        if symbol == self.spy:
            # Log daily summary
            equity = self.Portfolio.TotalPortfolioValue
            self.Log(f"EOD Equity: ${equity:,.2f}, Day P/L: ${self._daily_pnl:,.2f}")

    def OnEndOfAlgorithm(self) -> None:
        """Final algorithm cleanup and reporting."""
        self.Log("=" * 50)
        self.Log("FINAL REPORT")
        self.Log("=" * 50)
        self.Log(f"Final Equity: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Log(f"Circuit Breaker Status: {self.circuit_breaker.get_status()}")
        if self.smart_executor:
            stats = self.smart_executor.get_statistics()
            self.Log(f"Execution Stats: {stats}")

        # Resource usage statistics
        if hasattr(self, "resource_monitor"):
            resource_stats = self.resource_monitor.get_statistics()
            self.Log("=" * 50)
            self.Log("RESOURCE USAGE STATISTICS")
            self.Log("=" * 50)
            if "memory" in resource_stats:
                self.Log("Memory Usage:")
                self.Log(f"  Min: {resource_stats['memory']['min_pct']:.1f}%")
                self.Log(f"  Max: {resource_stats['memory']['max_pct']:.1f}%")
                self.Log(f"  Avg: {resource_stats['memory']['avg_pct']:.1f}%")
            if "cpu" in resource_stats:
                self.Log("CPU Usage:")
                self.Log(f"  Min: {resource_stats['cpu']['min_pct']:.1f}%")
                self.Log(f"  Max: {resource_stats['cpu']['max_pct']:.1f}%")
                self.Log(f"  Avg: {resource_stats['cpu']['avg_pct']:.1f}%")
            if "alerts" in resource_stats:
                self.Log("Resource Alerts:")
                self.Log(f"  Total: {resource_stats['alerts']['total']}")
                self.Log(f"  Critical: {resource_stats['alerts']['critical']}")
                self.Log(f"  Warnings: {resource_stats['alerts']['warning']}")

        # Object Store statistics
        if hasattr(self, "storage_monitor") and self.storage_monitor:
            storage_stats = self.storage_monitor.get_statistics()
            self.Log("=" * 50)
            self.Log("OBJECT STORE STATISTICS")
            self.Log("=" * 50)
            self.Log(
                f"Storage Used: {storage_stats['current_usage_gb']:.2f}GB / {storage_stats['limit_gb']}GB ({storage_stats['current_usage_pct']:.1f}%)"
            )
            self.Log(f"Files: {storage_stats['file_count']:,} / {storage_stats['file_limit']:,}")
            if storage_stats.get("growth_rate_gb_per_day"):
                self.Log(f"Growth Rate: {storage_stats['growth_rate_gb_per_day']:.3f}GB/day")
            if storage_stats.get("days_until_full"):
                self.Log(f"Days Until Full: {storage_stats['days_until_full']}")
            if storage_stats.get("by_category"):
                self.Log("Storage by Category:")
                for category, stats in storage_stats["by_category"].items():
                    self.Log(f"  {category}: {stats['size_mb']:.1f}MB ({stats['count']} files)")


class QuantConnectOptionsBot(OptionsTradingBot):
    """
    Alias for deployment compatibility.
    """

    pass


# Convenience function for local testing
def create_algorithm() -> OptionsTradingBot:
    """Create algorithm instance for testing."""
    return OptionsTradingBot()
