"""
Hybrid Options Trading Bot - Main QuantConnect Algorithm

A semi-autonomous options trading system with:
- Autonomous options strategies (37+ QuantConnect OptionStrategies)
- Manual order submission via UI/API
- Bot-managed positions with automated profit-taking
- Recurring order templates
- Unified position tracking across all sources
- LLM sentiment integration (UPGRADE-014, December 2025)

Architecture:
- 3 Order Sources: Autonomous + Manual + Recurring
- 2 Executors: OptionStrategiesExecutor + ManualLegsExecutor
- 1 Position Manager: BotManagedPositions
- 1 API Interface: OrderQueueAPI
- Risk Management: CircuitBreaker + RiskManager + SentimentFilter

This algorithm integrates all hybrid architecture modules.
- Nov 30, 2025: Hybrid architecture modules
- Dec 1, 2025: LLM sentiment integration (UPGRADE-014)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any


# QuantConnect imports (available at runtime)
try:
    from AlgorithmImports import *
except ImportError:
    # Stubs for development/testing
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


# Local imports
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.base_options_bot import BaseOptionsBot, CheckIntervals

from api import OrderQueueAPI
from execution import (
    create_bot_position_manager,
    create_manual_legs_executor,
    create_option_strategies_executor,
    create_recurring_order_manager,
)

# LLM Sentiment Integration (UPGRADE-014)
from llm import (
    FilterDecision,
    FilterResult,
    NewsAlertConfig,
    NewsAlertManager,
    TradingConstraints,
    create_ensemble,
    create_llm_guardrails,
    create_sentiment_filter,
    create_signal_from_ensemble,
)
from utils.object_store import (
    StorageCategory,
    create_sentiment_persistence,
)


class HybridOptionsBot(BaseOptionsBot):
    """
    Semi-autonomous hybrid options trading algorithm.

    Combines:
    1. Autonomous trading (OptionStrategies based on IV Rank)
    2. Manual orders (from UI via API)
    3. Recurring templates (scheduled orders)
    4. Bot-managed positions (automated profit-taking/rolling)

    All positions tracked uniformly regardless of source.

    Extends BaseOptionsBot which provides:
    - Configuration loading
    - Risk management (RiskLimits, RiskManager, CircuitBreaker)
    - Resource monitoring
    - Object store persistence
    """

    def _setup_basic(self) -> None:
        """Override base setup for hybrid-specific dates."""
        self.SetStartDate(2024, 11, 1)  # Conservative 1-month backtest
        self.SetEndDate(2024, 11, 30)
        self.SetCash(100000)

        # CRITICAL: Charles Schwab allows ONLY ONE algorithm per account
        self.SetBrokerageModel(BrokerageName.CharlesSchwab, AccountType.Margin)

    def _setup_strategy_specific(self) -> None:
        """Initialize hybrid architecture components."""
        # =====================================================================
        # LLM SENTIMENT INTEGRATION (UPGRADE-014)
        # =====================================================================
        self._setup_sentiment_components()

        # =====================================================================
        # HYBRID ARCHITECTURE - EXECUTORS
        # =====================================================================

        # 1. OptionStrategies Executor (Autonomous - 37+ strategies)
        options_exec_config = self._get_config("option_strategies_executor", {})
        if options_exec_config.get("enabled", True):
            self.options_executor = create_option_strategies_executor(
                algorithm=self,
                config=options_exec_config,
            )
            self.Debug("✅ OptionStrategiesExecutor initialized (autonomous trading enabled)")
        else:
            self.options_executor = None
            self.Debug("⚠️  OptionStrategiesExecutor disabled")

        # 2. Manual Legs Executor (Two-part spread strategy)
        manual_exec_config = self._get_config("manual_legs_executor", {})
        if manual_exec_config.get("enabled", True):
            self.manual_executor = create_manual_legs_executor(
                algorithm=self,
                config=manual_exec_config,
            )
            self.Debug("✅ ManualLegsExecutor initialized (two-part spreads enabled)")
        else:
            self.manual_executor = None
            self.Debug("⚠️  ManualLegsExecutor disabled")

        # =====================================================================
        # HYBRID ARCHITECTURE - POSITION MANAGEMENT
        # =====================================================================

        # Bot-managed positions (automated profit-taking)
        bot_manager_config = self._get_config("bot_managed_positions", {})
        if bot_manager_config.get("enabled", True):
            self.bot_manager = create_bot_position_manager(
                algorithm=self,
                config=bot_manager_config,
            )
            self.Debug("✅ BotManagedPositions initialized (auto profit-taking enabled)")
        else:
            self.bot_manager = None
            self.Debug("⚠️  BotManagedPositions disabled")

        # =====================================================================
        # HYBRID ARCHITECTURE - ORDER SOURCES
        # =====================================================================

        # Order Queue API (for UI and manual orders)
        self.order_queue = OrderQueueAPI(algorithm=self)
        self.Debug("✅ OrderQueueAPI initialized (manual orders enabled)")

        # Recurring Order Manager (scheduled templates)
        recurring_config = self._get_config("recurring_order_manager", {})
        if recurring_config.get("enabled", True):
            self.recurring_manager = create_recurring_order_manager(
                algorithm=self,
                config=recurring_config,
            )
            self.Debug("✅ RecurringOrderManager initialized (scheduled orders enabled)")
        else:
            self.recurring_manager = None
            self.Debug("⚠️  RecurringOrderManager disabled")

        # =====================================================================
        # DATA SUBSCRIPTIONS
        # =====================================================================
        self._setup_universe()

        # =====================================================================
        # SCHEDULED TASKS
        # =====================================================================
        self._setup_schedules()

        # =====================================================================
        # TRACKING
        # =====================================================================
        self._last_strategy_check_time = self.Time
        self._last_recurring_check_time = self.Time
        self._last_cb_log_time = self.Time  # Circuit breaker log throttle
        self._last_resource_check = self.Time
        self._position_count_by_source = defaultdict(int)
        self._daily_pnl = 0.0
        self._peak_equity = self.Portfolio.TotalPortfolioValue

        # Greeks use IV and require no warmup (LEAN PR #6720).
        # Uncomment below only if adding technical indicators that need history:
        # self.SetWarmUp(timedelta(days=30))

        # =====================================================================
        # INITIALIZATION COMPLETE
        # =====================================================================
        self._check_resources()

        # Condense component status into summary lines
        components = {
            "Autonomous": self.options_executor,
            "Manual": self.manual_executor,
            "BotMgr": self.bot_manager,
            "Recurring": self.recurring_manager,
            "ObjStore": self.object_store_manager,
            "Sentiment": self.sentiment_filter,
            "NewsAlerts": self.news_alert_manager,
            "Guardrails": self.llm_guardrails,
        }
        enabled = [k for k, v in components.items() if v is not None]
        disabled = [k for k, v in components.items() if v is None]

        self.Debug("=" * 60)
        self.Debug("HYBRID OPTIONS BOT INITIALIZED")
        self.Debug(f"  Enabled: {', '.join(enabled) if enabled else 'None'}")
        if disabled:
            self.Debug(f"  Disabled: {', '.join(disabled)}")
        self.Debug("=" * 60)

    def OnData(self, slice: Slice) -> None:
        """
        Process market data and execute hybrid trading logic.

        Order of operations:
        1. Check circuit breaker status
        2. Update sentiment signals (UPGRADE-014)
        3. Process queued manual orders (with sentiment validation)
        4. Run autonomous strategies (if enabled and sentiment favorable)
        5. Update bot-managed positions
        6. Check recurring templates
        7. Monitor resources
        """
        # Skip if warming up
        if self.IsWarmingUp:
            return

        # =================================================================
        # 1. CIRCUIT BREAKER CHECK
        # =================================================================
        if not self.circuit_breaker.can_trade():
            # Trading halted by circuit breaker - log once per hour
            if (self.Time - self._last_cb_log_time).total_seconds() > CheckIntervals.CIRCUIT_BREAKER_LOG:
                self.Debug("⚠️  Trading halted by circuit breaker")
                self._last_cb_log_time = self.Time
            return

        # =================================================================
        # 2. UPDATE SENTIMENT SIGNALS (UPGRADE-014)
        # =================================================================
        self._update_sentiment_signals(slice)

        # =================================================================
        # 3. PROCESS QUEUED MANUAL ORDERS (with sentiment validation)
        # =================================================================
        self._process_order_queue(slice)

        # =================================================================
        # 4. RUN AUTONOMOUS STRATEGIES (with sentiment check)
        # =================================================================
        if self.options_executor and self._should_check_strategies():
            self._run_autonomous_strategies(slice)

        # =================================================================
        # 5. UPDATE BOT-MANAGED POSITIONS
        # =================================================================
        if self.bot_manager:
            self._update_bot_positions(slice)

        # =================================================================
        # 6. CHECK RECURRING TEMPLATES
        # =================================================================
        if self.recurring_manager and self._should_check_recurring():
            self._check_recurring_orders(slice)

        # =================================================================
        # 7. MONITOR RESOURCES (every 30 seconds)
        # =================================================================
        if (self.Time - self._last_resource_check).total_seconds() > CheckIntervals.RESOURCE:
            self._check_resources()
            self._last_resource_check = self.Time

    def _process_order_queue(self, slice: Slice) -> None:
        """Process pending manual orders from queue."""
        pending_orders = self.order_queue.get_pending_orders()

        if not pending_orders:
            return

        self.Debug(f"📥 Processing {len(pending_orders)} queued order(s)")

        for order_request in pending_orders:
            try:
                # Validate risk before execution
                if not self._check_risk_limits(order_request):
                    self.order_queue.mark_order_rejected(order_request.order_id, "Risk limits exceeded")
                    continue

                # Execute via appropriate executor
                if order_request.execution_type == "option_strategy":
                    # Use OptionStrategiesExecutor for predefined strategies
                    if self.options_executor:
                        self.options_executor.execute_strategy_order(order_request, slice)
                        self.order_queue.mark_order_processing(order_request.order_id)
                    else:
                        self.order_queue.mark_order_rejected(
                            order_request.order_id, "OptionStrategiesExecutor disabled"
                        )

                elif order_request.execution_type == "manual_legs":
                    # Use ManualLegsExecutor for custom spreads
                    if self.manual_executor:
                        self.manual_executor.execute_manual_order(order_request, slice)
                        self.order_queue.mark_order_processing(order_request.order_id)
                    else:
                        self.order_queue.mark_order_rejected(order_request.order_id, "ManualLegsExecutor disabled")

                else:
                    self.order_queue.mark_order_rejected(
                        order_request.order_id, f"Unknown execution type: {order_request.execution_type}"
                    )

            except Exception as e:
                self.Debug(f"❌ Error processing order {order_request.order_id}: {e}")
                self.order_queue.mark_order_rejected(order_request.order_id, str(e))

    def _run_autonomous_strategies(self, slice: Slice) -> None:
        """Run autonomous options strategies based on IV Rank."""
        if not self.options_executor:
            return

        try:
            # Let OptionStrategiesExecutor handle autonomous logic
            self.options_executor.on_data(slice)
        except Exception as e:
            self.Debug(f"❌ Error in autonomous strategies: {e}")

    def _update_bot_positions(self, slice: Slice) -> None:
        """Update bot-managed positions with profit-taking and rolling."""
        if not self.bot_manager:
            return

        try:
            # Let BotManagedPositions handle position management
            self.bot_manager.on_data(slice)
        except Exception as e:
            self.Debug(f"❌ Error updating bot positions: {e}")

    def _check_recurring_orders(self, slice: Slice) -> None:
        """Check if any recurring templates should execute."""
        if not self.recurring_manager:
            return

        try:
            # Let RecurringOrderManager handle scheduled checks
            orders_created = self.recurring_manager.check_templates(slice)

            if orders_created:
                self.Debug(f"📅 Recurring manager created {len(orders_created)} order(s)")

                # Add created orders to queue for execution
                for order in orders_created:
                    self.order_queue.submit_order(order)

        except Exception as e:
            self.Debug(f"❌ Error checking recurring orders: {e}")

    def _setup_universe(self) -> None:
        """Subscribe to options data for primary symbols."""
        # Primary symbols for options trading
        primary_symbols = self._get_config("option_strategies_executor", {}).get(
            "primary_symbols", ["SPY", "QQQ", "IWM"]
        )

        self.option_symbols = {}

        for ticker in primary_symbols:
            try:
                # Add equity
                equity = self.add_equity(ticker, Resolution.Minute)

                # Add options with filter
                option = self.add_option(ticker, Resolution.Minute)

                # Filter: ±20 strikes, 0-180 DTE (covers 30-180 DTE range)
                option.set_filter(-20, 20, 0, 180)

                self.option_symbols[equity.Symbol] = option.Symbol

                self.Debug(f"✅ Subscribed to {ticker} options")

            except Exception as e:
                self.Debug(f"❌ Failed to subscribe to {ticker}: {e}")

    def _setup_schedules(self) -> None:
        """Schedule recurring tasks.

        Note: Strategy and recurring checks are handled in OnData via
        _should_check_strategies() and _should_check_recurring() with
        their own timing logic. No scheduled backups needed.
        """
        # Daily risk review at market close
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.BeforeMarketClose("SPY", 5), self._daily_risk_review)

    def _daily_risk_review(self) -> None:
        """Daily review of risk metrics and positions."""
        try:
            # Calculate daily P&L
            current_equity = self.Portfolio.TotalPortfolioValue
            self._daily_pnl = (current_equity - self.risk_manager.starting_equity) / self.risk_manager.starting_equity

            # Update peak equity for drawdown calculation
            if current_equity > self._peak_equity:
                self._peak_equity = current_equity

            # Check circuit breaker conditions
            self.circuit_breaker.check_daily_loss(self._daily_pnl)
            self.circuit_breaker.check_drawdown(current_equity, self._peak_equity)

            # Log daily summary
            self.Debug("=" * 80)
            self.Debug(f"📊 DAILY SUMMARY - {self.Time.strftime('%Y-%m-%d')}")
            self.Debug(f"   Equity: ${current_equity:,.2f} (Daily P&L: {self._daily_pnl:+.2%})")
            self.Debug(f"   Open Positions: {len([h for h in self.Portfolio.Values if h.Invested])}")
            self.Debug(f"   Circuit Breaker: {'✅ Active' if self.circuit_breaker.can_trade() else '⚠️  HALTED'}")
            self.Debug("=" * 80)

        except Exception as e:
            self.Debug(f"❌ Error in daily risk review: {e}")

    def _check_risk_limits(self, order_request: Any) -> bool:
        """Check if order would violate risk limits."""
        try:
            # Get current portfolio value
            portfolio_value = self.Portfolio.TotalPortfolioValue

            # Estimate order value (rough approximation)
            # For ComboOrders, use net debit/credit
            estimated_value = abs(getattr(order_request, "net_debit", 0)) * getattr(order_request, "quantity", 1) * 100

            # Check position size limit
            position_size_pct = estimated_value / portfolio_value if portfolio_value > 0 else 0

            if position_size_pct > self.risk_limits.max_position_size:
                self.Debug(
                    f"⚠️  Order rejected: Position size {position_size_pct:.1%} exceeds limit {self.risk_limits.max_position_size:.1%}"
                )
                return False

            # Check max open positions
            max_positions = self._get_config("risk_management", {}).get("max_open_positions", 5)
            current_positions = len([h for h in self.Portfolio.Values if h.Invested])

            if current_positions >= max_positions:
                self.Debug(f"⚠️  Order rejected: Already at max positions ({max_positions})")
                return False

            return True

        except Exception as e:
            self.Debug(f"❌ Error checking risk limits: {e}")
            return False  # Reject on error (fail safe)

    def _should_check_strategies(self) -> bool:
        """Determine if it's time to check autonomous strategies (every 5 min)."""
        if (self.Time - self._last_strategy_check_time).total_seconds() >= CheckIntervals.STRATEGY:
            self._last_strategy_check_time = self.Time
            return True
        return False

    def _should_check_recurring(self) -> bool:
        """Determine if it's time to check recurring templates (every 1 hour)."""
        if (self.Time - self._last_recurring_check_time).total_seconds() >= CheckIntervals.RECURRING:
            self._last_recurring_check_time = self.Time
            return True
        return False

    def _check_resources(self) -> None:
        """Check resource usage and log warnings."""
        if self.resource_monitor:
            try:
                self.resource_monitor.check_resources()
            except Exception as e:
                self.Debug(f"❌ Error checking resources: {e}")

    # =========================================================================
    # SENTIMENT INTEGRATION METHODS (UPGRADE-014)
    # =========================================================================

    def _update_sentiment_signals(self, slice: Slice) -> None:
        """
        Update sentiment signals for all subscribed symbols.

        Analyzes news and generates sentiment signals that:
        - Feed into entry filtering (SentimentFilter)
        - Trigger circuit breaker on critical negative sentiment
        - Adjust position sizing via LLM guardrails
        """
        if not self.llm_ensemble:
            return  # LLM ensemble not enabled

        # Only check sentiment every 5 minutes to manage API costs
        if hasattr(self, "_last_sentiment_check"):
            if (self.Time - self._last_sentiment_check).total_seconds() < CheckIntervals.SENTIMENT:
                return

        self._last_sentiment_check = self.Time

        # Analyze sentiment for each symbol we're trading
        for equity_symbol in self.option_symbols.keys():
            try:
                ticker = str(equity_symbol.Value)

                # Get recent news (would integrate with news feed in production)
                # For now, use placeholder that would be replaced with actual news
                news_text = self._get_recent_news(ticker)

                if not news_text:
                    continue

                # Analyze with LLM ensemble
                result = self.llm_ensemble.analyze_sentiment(news_text)

                # Create sentiment signal for filter
                if self.sentiment_filter:
                    signal = create_signal_from_ensemble(
                        ensemble_result=result,
                        symbol=ticker,
                    )
                    self.sentiment_filter.add_signal(signal)

                # Track sentiment history
                self._sentiment_history[ticker].append(result.sentiment.score)

                # Keep only last 100 signals per symbol
                if len(self._sentiment_history[ticker]) > 100:
                    self._sentiment_history[ticker] = self._sentiment_history[ticker][-100:]

                # Check for sentiment-based circuit breaker trigger
                self._check_sentiment_circuit_breaker(ticker, result.sentiment.score)

                # Log significant sentiment changes
                if abs(result.sentiment.score) > 0.5:
                    self.Debug(
                        f"📊 {ticker} sentiment: {result.sentiment.score:+.2f} "
                        f"(confidence: {result.sentiment.confidence:.2f})"
                    )

            except Exception as e:
                self.Debug(f"⚠️  Error analyzing sentiment for {ticker}: {e}")

    def _check_sentiment_circuit_breaker(self, symbol: str, sentiment_score: float) -> None:
        """
        Check if sentiment should trigger circuit breaker.

        Triggers on:
        - Extremely negative sentiment (below threshold)
        - Consecutive negative sentiment readings
        - Critical news keywords detected
        """
        if self._sentiment_halt_threshold is None:
            return  # Sentiment circuit breaker disabled

        # Check for extreme negative sentiment
        if sentiment_score < self._sentiment_halt_threshold:
            self.circuit_breaker.halt_all_trading(f"Extreme negative sentiment for {symbol}: {sentiment_score:.2f}")
            return

        # Track consecutive negative sentiment
        if sentiment_score < 0:
            self._consecutive_negative_count += 1
            if self._consecutive_negative_count >= self._consecutive_negative_limit:
                self.circuit_breaker.halt_all_trading(
                    f"Consecutive negative sentiment ({self._consecutive_negative_count}x) for {symbol}"
                )
        else:
            self._consecutive_negative_count = 0  # Reset on positive sentiment

    def _get_recent_news(self, symbol: str) -> str | None:
        """
        Get recent news for symbol analysis using QuantConnect Tiingo data.

        Integrates with:
        - QuantConnect Tiingo News Data
        - Caches results to minimize API calls

        Args:
            symbol: Ticker symbol to get news for

        Returns:
            Combined news text for sentiment analysis, or None if no news
        """
        # Check news cache first (avoid excessive API calls)
        cache_key = f"{symbol}_{self.Time.date()}"
        if hasattr(self, "_news_cache") and cache_key in self._news_cache:
            cached = self._news_cache[cache_key]
            # Return cached result if less than 30 minutes old
            if (self.Time - cached["time"]).total_seconds() < 1800:
                return cached["text"]

        # Initialize news cache if needed
        if not hasattr(self, "_news_cache"):
            self._news_cache = {}

        try:
            # Get Tiingo news data from QuantConnect
            # Note: Requires TiingoNews data subscription
            ticker = symbol.upper()

            # Try to get news from Tiingo (if subscribed)
            if hasattr(self, "AddData"):
                # Check if we have news data for this symbol
                news_symbol = (
                    self.AddData(TiingoNews, ticker, Resolution.Daily).Symbol if "TiingoNews" in dir() else None
                )

                if news_symbol and news_symbol in self.Securities:
                    # Get recent news articles
                    history = (
                        self.History(TiingoNews, news_symbol, timedelta(days=1)) if "TiingoNews" in dir() else None
                    )

                    if history is not None and len(history) > 0:
                        # Combine headlines and descriptions
                        news_texts = []
                        for _, row in history.iterrows():
                            if hasattr(row, "title"):
                                news_texts.append(f"{row.title}. {getattr(row, 'description', '')}")

                        if news_texts:
                            combined_text = " ".join(news_texts[:5])  # Limit to 5 articles
                            self._news_cache[cache_key] = {"time": self.Time, "text": combined_text}
                            return combined_text

            # Fallback: No Tiingo subscription or no news found
            self._news_cache[cache_key] = {"time": self.Time, "text": None}
            return None

        except Exception as e:
            self.Debug(f"⚠️  Error fetching news for {symbol}: {e}")
            return None

    def _validate_sentiment_for_trade(
        self,
        symbol: str,
        direction: str,
        confidence: float = 0.5,
    ) -> FilterResult:
        """
        Validate if sentiment allows a trade.

        Args:
            symbol: Symbol to trade
            direction: 'long' or 'short'
            confidence: LLM confidence in the trade decision

        Returns:
            FilterResult with decision and reasoning
        """
        if not self.sentiment_filter:
            # No filter, allow all trades
            return FilterResult(
                decision=FilterDecision.ALLOW,
                reason=None,
                sentiment_score=0.0,
                confidence=0.0,
            )

        return self.sentiment_filter.check_entry(
            symbol=symbol,
            direction=direction,
        )

    def _validate_with_guardrails(
        self,
        action: str,
        symbol: str,
        position_size: float,
        confidence: float,
        sentiment_score: float,
    ) -> bool:
        """
        Validate trade decision with LLM guardrails.

        Args:
            action: 'buy' or 'sell'
            symbol: Symbol to trade
            position_size: Position size as fraction of portfolio
            confidence: Confidence in the trade
            sentiment_score: Current sentiment score

        Returns:
            True if trade is allowed, False otherwise
        """
        if not self.llm_guardrails:
            return True  # No guardrails, allow all

        result = self.llm_guardrails.validate_trade_decision(
            action=action,
            symbol=symbol,
            position_size=position_size,
            confidence=confidence,
            sentiment_score=sentiment_score,
        )

        if not result.passed:
            self.Debug(f"⚠️  Guardrails blocked trade: {symbol} - {result.violations}")
            return False

        return True

    def _get_sentiment_adjusted_size(
        self,
        base_size: float,
        symbol: str,
        confidence: float,
    ) -> float:
        """
        Adjust position size based on sentiment confidence.

        High confidence = full size
        Low confidence = reduced size
        Very low confidence = minimum size

        Args:
            base_size: Base position size as fraction of portfolio
            symbol: Symbol being traded
            confidence: LLM confidence score

        Returns:
            Adjusted position size
        """
        if not self.llm_guardrails:
            return base_size

        # Get scaling config
        guardrails_config = self._get_config("llm_guardrails", {})
        scaling_config = guardrails_config.get("confidence_scaling", {})

        if not scaling_config.get("enabled", False):
            return base_size

        low_threshold = scaling_config.get("low_confidence_threshold", 0.4)
        high_threshold = scaling_config.get("high_confidence_threshold", 0.8)
        low_multiplier = scaling_config.get("low_confidence_size_multiplier", 0.5)
        high_multiplier = scaling_config.get("high_confidence_size_multiplier", 1.0)

        # Scale based on confidence
        if confidence < low_threshold:
            adjusted_size = base_size * low_multiplier
        elif confidence > high_threshold:
            adjusted_size = base_size * high_multiplier
        else:
            # Linear interpolation between low and high
            ratio = (confidence - low_threshold) / (high_threshold - low_threshold)
            multiplier = low_multiplier + ratio * (high_multiplier - low_multiplier)
            adjusted_size = base_size * multiplier

        self.Debug(
            f"📏 Size adjustment for {symbol}: {base_size:.2%} → {adjusted_size:.2%} " f"(confidence: {confidence:.2f})"
        )

        return adjusted_size

    def get_sentiment_summary(self, symbol: str | None = None) -> dict[str, Any]:
        """
        Get sentiment summary for reporting.

        Args:
            symbol: Specific symbol to get summary for, or None for all

        Returns:
            Dictionary with sentiment statistics
        """
        if symbol:
            history = self._sentiment_history.get(symbol, [])
            if not history:
                return {"symbol": symbol, "signals": 0, "average": 0.0}

            return {
                "symbol": symbol,
                "signals": len(history),
                "average": sum(history) / len(history),
                "latest": history[-1] if history else 0.0,
                "min": min(history),
                "max": max(history),
            }

        # Return summary for all symbols
        summaries = {}
        for sym, history in self._sentiment_history.items():
            if history:
                summaries[sym] = {
                    "signals": len(history),
                    "average": sum(history) / len(history),
                    "latest": history[-1],
                }

        return {
            "total_symbols": len(summaries),
            "symbols": summaries,
            "filter_enabled": self.sentiment_filter is not None,
            "guardrails_enabled": self.llm_guardrails is not None,
            "ensemble_enabled": self.llm_ensemble is not None,
        }

    def _on_circuit_breaker_alert(self, message: str, details: dict) -> None:
        """Handle circuit breaker alerts."""
        urgency = details.get("reason", "UNKNOWN")
        self.Debug(f"🚨 CIRCUIT BREAKER ALERT [{urgency}]: {message}")

        # Could integrate with notification system here
        # - Email alerts
        # - Discord/Slack webhooks
        # - SMS via Twilio
        # For now, just log

    def _setup_sentiment_components(self) -> None:
        """Initialize LLM sentiment integration components (UPGRADE-014)."""
        # Check if sentiment integration is enabled
        llm_config = self._get_config("llm_integration", {})
        sentiment_filter_config = self._get_config("sentiment_filter", {})
        news_alert_config = self._get_config("news_alert_manager", {})
        guardrails_config = self._get_config("llm_guardrails", {})

        # ---------------------------------------------------------------
        # LLM Ensemble (for sentiment analysis)
        # ---------------------------------------------------------------
        if llm_config.get("enabled", False):
            try:
                self.llm_ensemble = create_ensemble(llm_config)
                self.Debug("✅ LLM Ensemble initialized (FinBERT + GPT-4o + Claude)")
            except Exception as e:
                self.llm_ensemble = None
                self.Debug(f"⚠️  LLM Ensemble failed to initialize: {e}")
        else:
            self.llm_ensemble = None
            self.Debug("⚠️  LLM Ensemble disabled")

        # ---------------------------------------------------------------
        # Sentiment Filter (blocks trades based on sentiment)
        # ---------------------------------------------------------------
        if sentiment_filter_config.get("enabled", False):
            try:
                self.sentiment_filter = create_sentiment_filter(
                    min_sentiment_for_long=sentiment_filter_config.get("min_sentiment_for_long", 0.0),
                    max_sentiment_for_short=sentiment_filter_config.get("max_sentiment_for_short", 0.0),
                    min_confidence=sentiment_filter_config.get("min_confidence", 0.5),
                    lookback_hours=sentiment_filter_config.get("lookback_hours", 24),
                )
                self.Debug("✅ Sentiment Filter initialized (entry filtering enabled)")
            except Exception as e:
                self.sentiment_filter = None
                self.Debug(f"⚠️  Sentiment Filter failed to initialize: {e}")
        else:
            self.sentiment_filter = None
            self.Debug("⚠️  Sentiment Filter disabled")

        # ---------------------------------------------------------------
        # News Alert Manager (detects critical news events)
        # ---------------------------------------------------------------
        if news_alert_config.get("enabled", False):
            try:
                alert_config = NewsAlertConfig(
                    high_impact_sentiment_threshold=news_alert_config.get("impact_thresholds", {}).get("high", 0.6),
                    circuit_breaker_sentiment_threshold=news_alert_config.get("impact_thresholds", {}).get(
                        "critical", 0.8
                    ),
                    enable_circuit_breaker_triggers=news_alert_config.get("trigger_circuit_breaker_on_critical", True),
                )
                self.news_alert_manager = NewsAlertManager(config=alert_config)
                self.Debug("✅ News Alert Manager initialized (critical news detection enabled)")
            except Exception as e:
                self.news_alert_manager = None
                self.Debug(f"⚠️  News Alert Manager failed to initialize: {e}")
        else:
            self.news_alert_manager = None
            self.Debug("⚠️  News Alert Manager disabled")

        # ---------------------------------------------------------------
        # LLM Guardrails (validates trading decisions)
        # ---------------------------------------------------------------
        if guardrails_config.get("enabled", False):
            try:
                constraints = TradingConstraints(
                    max_position_size_pct=guardrails_config.get("max_position_size", 0.25),
                    min_confidence_for_trade=guardrails_config.get("min_confidence_for_trade", 0.5),
                    blocked_symbols=guardrails_config.get("blocked_symbols", []),
                    max_daily_trades=guardrails_config.get("max_daily_trades", 50),
                )
                self.llm_guardrails = create_llm_guardrails(constraints=constraints)
                self.Debug("✅ LLM Guardrails initialized (trading safety enabled)")
            except Exception as e:
                self.llm_guardrails = None
                self.Debug(f"⚠️  LLM Guardrails failed to initialize: {e}")
        else:
            self.llm_guardrails = None
            self.Debug("⚠️  LLM Guardrails disabled")

        # ---------------------------------------------------------------
        # Connect Sentiment to Circuit Breaker
        # ---------------------------------------------------------------
        sentiment_breaker_config = self._get_config("sentiment_circuit_breaker", {})
        if sentiment_breaker_config.get("enabled", False):
            # Store sentiment thresholds for circuit breaker integration
            self._sentiment_halt_threshold = sentiment_breaker_config.get("sentiment_halt_threshold", -0.7)
            self._consecutive_negative_limit = sentiment_breaker_config.get("consecutive_negative_sentiment", 5)
            self._critical_news_keywords = sentiment_breaker_config.get("critical_news_keywords", [])
            self._consecutive_negative_count = 0
            self.Debug(f"✅ Sentiment Circuit Breaker enabled (halt at {self._sentiment_halt_threshold})")
        else:
            self._sentiment_halt_threshold = None
            self.Debug("⚠️  Sentiment Circuit Breaker disabled")

        # Track sentiment history
        self._sentiment_history: dict[str, list[float]] = defaultdict(list)

        # ---------------------------------------------------------------
        # Sentiment Persistence (UPGRADE-014)
        # ---------------------------------------------------------------
        if hasattr(self, "object_store_manager") and self.object_store_manager:
            try:
                self.sentiment_persistence = create_sentiment_persistence(self.object_store_manager)
                # Load persisted sentiment history
                persisted_history = self.sentiment_persistence.get_all_sentiment_history()
                for symbol, scores in persisted_history.items():
                    self._sentiment_history[symbol] = scores
                self.Debug(f"✅ Sentiment Persistence initialized ({len(persisted_history)} symbols loaded)")
            except Exception as e:
                self.sentiment_persistence = None
                self.Debug(f"⚠️  Sentiment Persistence failed to initialize: {e}")
        else:
            self.sentiment_persistence = None
            self.Debug("⚠️  Sentiment Persistence disabled (Object Store not available)")

    def _get_node_info(self) -> str:
        """Get current compute node information."""
        qc_config = self._get_config("quantconnect", {})
        nodes = qc_config.get("compute_nodes", {})

        # Determine which node we're running on
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

    def OnOrderEvent(self, orderEvent: OrderEvent) -> None:
        """Handle order events (fills, cancellations, etc.)."""
        order = self.Transactions.GetOrderById(orderEvent.OrderId)

        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"✅ Order filled: {order.Symbol} ({orderEvent.FillQuantity} @ ${orderEvent.FillPrice:.2f})")

            # Update bot manager if this is a bot-managed position
            if self.bot_manager:
                self.bot_manager.on_order_filled(orderEvent)

            # Track for circuit breaker
            # (Will implement P&L tracking in future iteration)

        elif orderEvent.Status == OrderStatus.Canceled:
            self.Debug(f"⚠️  Order canceled: {order.Symbol}")

        elif orderEvent.Status == OrderStatus.Invalid:
            self.Debug(f"❌ Order invalid: {order.Symbol} - {orderEvent.Message}")

    def OnEndOfAlgorithm(self) -> None:
        """Cleanup and final reporting."""
        self.Debug("=" * 80)
        self.Debug("📊 ALGORITHM COMPLETED")
        self.Debug(f"   Final Equity: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Debug(
            f"   Total Return: {(self.Portfolio.TotalPortfolioValue / self.risk_manager.starting_equity - 1):+.2%}"
        )
        self.Debug(f"   Peak Equity: ${self._peak_equity:,.2f}")

        # Circuit breaker status
        if self.circuit_breaker.is_halted:
            self.Debug("   ⚠️  Trading was HALTED by circuit breaker")
            self.Debug(f"   Halt Reason: {self.circuit_breaker.halt_reason}")
        else:
            self.Debug("   ✅ Circuit breaker: No halts")

        # =====================================================================
        # PERSIST SENTIMENT DATA (UPGRADE-014)
        # =====================================================================
        self._persist_sentiment_data()

        # =====================================================================
        # SENTIMENT SUMMARY
        # =====================================================================
        if self._sentiment_history:
            self.Debug("📊 SENTIMENT SUMMARY")
            total_signals = sum(len(h) for h in self._sentiment_history.values())
            self.Debug(f"   Total Signals: {total_signals}")
            self.Debug(f"   Symbols Tracked: {len(self._sentiment_history)}")
            for symbol, history in list(self._sentiment_history.items())[:5]:
                if history:
                    avg = sum(history) / len(history)
                    self.Debug(f"   {symbol}: avg={avg:+.2f}, signals={len(history)}")

        self.Debug("=" * 80)

    def _persist_sentiment_data(self) -> None:
        """
        Persist sentiment data to Object Store on algorithm shutdown.

        Saves:
        - Sentiment history per symbol
        - Provider performance metrics (for dynamic weighting)
        - LLM ensemble weights (for cross-session learning)
        """
        if not self.sentiment_persistence:
            return

        try:
            # Save sentiment history for each symbol
            saved_count = 0
            for symbol, history in self._sentiment_history.items():
                if history:
                    self.sentiment_persistence.save_sentiment_history(symbol, history)
                    saved_count += 1

            self.Debug(f"✅ Persisted sentiment for {saved_count} symbols")

            # Save LLM ensemble weights for cross-session learning
            if self.llm_ensemble and hasattr(self.llm_ensemble, "weights"):
                provider_perf = self.llm_ensemble.get_provider_performance()
                if provider_perf:
                    self.sentiment_persistence.save_provider_performance(provider_perf)
                    self.Debug("✅ Persisted LLM provider performance")

            # Save sentiment filter stats
            if self.sentiment_filter:
                stats = self.sentiment_filter.get_stats()
                self.object_store_manager.save(
                    key="sentiment_filter_stats",
                    data=stats,
                    category=StorageCategory.SENTIMENT_DATA,
                    expire_days=30,
                )
                self.Debug(f"✅ Persisted filter stats: {stats.get('total_checks', 0)} checks")

        except Exception as e:
            self.Debug(f"⚠️  Error persisting sentiment data: {e}")
