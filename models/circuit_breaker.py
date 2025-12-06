#!/usr/bin/env python3
"""
Trading Circuit Breaker for QuantConnect Trading Bot

This module provides circuit breaker functionality to automatically halt trading
when risk thresholds are breached. This is a critical safety mechanism for
autonomous trading systems.

Author: QuantConnect Trading Bot
Date: 2025-11-25
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


# Get module logger - configuration should be done at application level
logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """State of the circuit breaker."""

    CLOSED = "closed"  # Normal operation, trading allowed
    OPEN = "open"  # Tripped, trading halted
    HALF_OPEN = "half_open"  # Testing if conditions have normalized


class TripReason(Enum):
    """Reason for circuit breaker trip."""

    DAILY_LOSS = "daily_loss_exceeded"
    MAX_DRAWDOWN = "max_drawdown_exceeded"
    CONSECUTIVE_LOSSES = "consecutive_losses_exceeded"
    VOLATILITY_SPIKE = "volatility_spike"
    POSITION_LIMIT = "position_limit_exceeded"
    MANUAL_HALT = "manual_halt"
    SYSTEM_ERROR = "system_error"
    # UPGRADE-014: Sentiment-based triggers
    NEGATIVE_SENTIMENT = "negative_sentiment_spike"
    CRITICAL_NEWS = "critical_news_event"
    SENTIMENT_DIVERGENCE = "sentiment_divergence"
    # UPGRADE-010: Anomaly detection trigger
    ANOMALY_DETECTED = "anomaly_detected"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker thresholds."""

    # Loss thresholds
    max_daily_loss_pct: float = 0.03  # 3% daily loss limit
    max_drawdown_pct: float = 0.10  # 10% max drawdown
    max_consecutive_losses: int = 5  # 5 consecutive losing trades

    # Volatility thresholds
    max_volatility_multiple: float = 3.0  # 3x normal volatility

    # Position thresholds
    max_position_pct: float = 0.25  # 25% max position size

    # Recovery settings
    cooldown_minutes: int = 30  # Minimum time before reset
    require_human_reset: bool = True  # Require human intervention to reset

    # UPGRADE-014: Sentiment thresholds
    sentiment_halt_threshold: float = -0.8  # Halt on extreme negative sentiment
    consecutive_negative_sentiment: int = 3  # Halt after N consecutive negative signals
    critical_news_keywords: list[str] = field(
        default_factory=lambda: [
            "bankruptcy",
            "fraud",
            "sec investigation",
            "delisting",
            "halt",
            "suspended",
            "crash",
            "collapse",
            "scandal",
        ]
    )
    enable_sentiment_triggers: bool = True  # Enable sentiment-based halts


@dataclass
class CircuitBreakerEvent:
    """Record of a circuit breaker event."""

    timestamp: datetime
    state: CircuitBreakerState
    reason: TripReason | None = None
    details: dict = field(default_factory=dict)
    resolved: bool = False
    resolved_by: str | None = None


class TradingCircuitBreaker:
    """
    Circuit breaker for trading risk management.

    Automatically halts trading when critical thresholds are breached.
    Requires human intervention to reset by default.

    Example usage:
        breaker = TradingCircuitBreaker()

        # In your trading loop:
        if not breaker.can_trade():
            return  # Trading halted

        # Check specific conditions
        breaker.check_daily_loss(portfolio)
        breaker.check_drawdown(portfolio)
    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        alert_callback: Callable[[str, dict], None] | None = None,
        log_file: Path | None = None,
    ):
        """
        Initialize the circuit breaker.

        Args:
            config: Circuit breaker configuration
            alert_callback: Function to call when circuit breaker trips
            log_file: Path to log file for audit trail (NOT USED in QuantConnect cloud)
        """
        self.config = config or CircuitBreakerConfig()
        self.alert_callback = alert_callback
        self.log_file = log_file or Path("circuit_breaker_log.json")

        self._state = CircuitBreakerState.CLOSED
        self._trip_reason: TripReason | None = None
        self._trip_time: datetime | None = None
        self._events: list[CircuitBreakerEvent] = []
        self._consecutive_losses = 0
        self._daily_pnl = 0.0
        self._peak_equity = 0.0

        # UPGRADE-014: Sentiment tracking
        self._consecutive_negative_sentiment: dict[str, int] = {}  # Per-symbol tracking
        self._sentiment_history: list[dict] = []  # Recent sentiment signals
        self._max_sentiment_history = 50

        # QuantConnect compatibility: track if running in QC cloud
        self._algorithm = None  # Will be set when integrated with QC

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is tripped (trading halted)."""
        return self._state == CircuitBreakerState.OPEN

    def can_trade(self) -> bool:
        """
        Check if trading is currently allowed.

        Returns:
            True if trading is allowed, False if halted
        """
        return self._state == CircuitBreakerState.CLOSED

    def check(self, portfolio: dict) -> tuple[bool, str | None]:
        """
        Run all circuit breaker checks.

        Args:
            portfolio: Portfolio state dictionary with keys:
                - daily_pnl_pct: Daily P&L percentage
                - current_equity: Current portfolio value
                - peak_equity: Peak portfolio value
                - positions: Dict of position info

        Returns:
            Tuple of (can_trade, reason_if_halted)
        """
        # Check daily loss
        daily_pnl = portfolio.get("daily_pnl_pct", 0.0)
        if daily_pnl < -self.config.max_daily_loss_pct:
            self._trip(
                TripReason.DAILY_LOSS,
                {"daily_pnl_pct": daily_pnl, "limit": self.config.max_daily_loss_pct},
            )
            return False, f"Daily loss limit breached: {daily_pnl:.2%}"

        # Check drawdown
        current_equity = portfolio.get("current_equity", 0)
        peak_equity = portfolio.get("peak_equity", current_equity)
        if peak_equity > 0:
            drawdown = (peak_equity - current_equity) / peak_equity
            if drawdown > self.config.max_drawdown_pct:
                self._trip(
                    TripReason.MAX_DRAWDOWN,
                    {"drawdown": drawdown, "limit": self.config.max_drawdown_pct},
                )
                return False, f"Max drawdown exceeded: {drawdown:.2%}"

        # Check consecutive losses
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self._trip(
                TripReason.CONSECUTIVE_LOSSES,
                {
                    "consecutive_losses": self._consecutive_losses,
                    "limit": self.config.max_consecutive_losses,
                },
            )
            return False, f"Consecutive losses exceeded: {self._consecutive_losses}"

        return True, None

    def check_daily_loss(self, daily_pnl_pct: float) -> bool:
        """
        Check if daily loss limit has been breached.

        Args:
            daily_pnl_pct: Daily P&L as percentage (negative for loss)

        Returns:
            True if OK, False if breached
        """
        if daily_pnl_pct <= -self.config.max_daily_loss_pct:
            self._trip(
                TripReason.DAILY_LOSS,
                {"daily_pnl_pct": daily_pnl_pct, "limit": self.config.max_daily_loss_pct},
            )
            return False
        return True

    def check_drawdown(self, current_equity: float, peak_equity: float) -> bool:
        """
        Check if max drawdown has been breached.

        Args:
            current_equity: Current portfolio value
            peak_equity: Peak portfolio value

        Returns:
            True if OK, False if breached
        """
        if peak_equity <= 0:
            return True

        drawdown = (peak_equity - current_equity) / peak_equity
        if drawdown > self.config.max_drawdown_pct:
            self._trip(
                TripReason.MAX_DRAWDOWN,
                {"drawdown": drawdown, "limit": self.config.max_drawdown_pct},
            )
            return False
        return True

    def record_trade_result(self, is_winner: bool) -> None:
        """
        Record a trade result for consecutive loss tracking.

        Args:
            is_winner: True if trade was profitable
        """
        if is_winner:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.config.max_consecutive_losses:
                self._trip(
                    TripReason.CONSECUTIVE_LOSSES,
                    {
                        "consecutive_losses": self._consecutive_losses,
                        "limit": self.config.max_consecutive_losses,
                    },
                )

    # =========================================================================
    # UPGRADE-014: Sentiment-Based Circuit Breaker Methods
    # =========================================================================

    def check_sentiment(
        self,
        symbol: str,
        sentiment_score: float,
        confidence: float = 0.5,
        source: str = "unknown",
    ) -> bool:
        """
        Check if sentiment triggers circuit breaker (UPGRADE-014).

        Args:
            symbol: Trading symbol
            sentiment_score: Sentiment score (-1.0 to 1.0)
            confidence: Confidence level of sentiment analysis
            source: Source of sentiment (finbert, ensemble, etc.)

        Returns:
            True if OK, False if circuit breaker triggered
        """
        if not self.config.enable_sentiment_triggers:
            return True

        # Record sentiment for history
        self._record_sentiment(symbol, sentiment_score, confidence, source)

        # Check for extreme negative sentiment
        if sentiment_score <= self.config.sentiment_halt_threshold:
            if confidence >= 0.6:  # Only trigger on confident signals
                self._trip(
                    TripReason.NEGATIVE_SENTIMENT,
                    {
                        "symbol": symbol,
                        "sentiment_score": sentiment_score,
                        "confidence": confidence,
                        "source": source,
                        "threshold": self.config.sentiment_halt_threshold,
                    },
                )
                return False

        # Track consecutive negative sentiment per symbol
        if sentiment_score < -0.2:  # Moderately negative
            self._consecutive_negative_sentiment[symbol] = self._consecutive_negative_sentiment.get(symbol, 0) + 1

            if self._consecutive_negative_sentiment[symbol] >= self.config.consecutive_negative_sentiment:
                self._trip(
                    TripReason.NEGATIVE_SENTIMENT,
                    {
                        "symbol": symbol,
                        "consecutive_negative": self._consecutive_negative_sentiment[symbol],
                        "threshold": self.config.consecutive_negative_sentiment,
                    },
                )
                return False
        else:
            # Reset on neutral/positive sentiment
            self._consecutive_negative_sentiment[symbol] = 0

        return True

    def check_news_event(
        self,
        symbol: str,
        headline: str,
        sentiment_score: float,
        impact: str = "medium",
    ) -> bool:
        """
        Check if news event should trigger circuit breaker (UPGRADE-014).

        Args:
            symbol: Trading symbol
            headline: News headline
            sentiment_score: Sentiment score of the news
            impact: Impact level ("low", "medium", "high", "critical")

        Returns:
            True if OK, False if circuit breaker triggered
        """
        if not self.config.enable_sentiment_triggers:
            return True

        headline_lower = headline.lower()

        # Check for critical keywords
        critical_match = any(kw in headline_lower for kw in self.config.critical_news_keywords)

        if critical_match and sentiment_score < -0.3:
            self._trip(
                TripReason.CRITICAL_NEWS,
                {
                    "symbol": symbol,
                    "headline": headline[:100],
                    "sentiment_score": sentiment_score,
                    "impact": impact,
                    "matched_keywords": [kw for kw in self.config.critical_news_keywords if kw in headline_lower],
                },
            )
            return False

        # High impact + very negative sentiment
        if impact == "critical" and sentiment_score < -0.5:
            self._trip(
                TripReason.CRITICAL_NEWS,
                {
                    "symbol": symbol,
                    "headline": headline[:100],
                    "sentiment_score": sentiment_score,
                    "impact": impact,
                },
            )
            return False

        return True

    def check_sentiment_divergence(
        self,
        symbol: str,
        market_direction: str,
        sentiment_direction: str,
        divergence_magnitude: float,
    ) -> bool:
        """
        Check for significant sentiment-price divergence (UPGRADE-014).

        Triggers when market moves opposite to sentiment with high magnitude.

        Args:
            symbol: Trading symbol
            market_direction: "bullish" or "bearish"
            sentiment_direction: "bullish" or "bearish"
            divergence_magnitude: How strong the divergence is (0-1)

        Returns:
            True if OK, False if circuit breaker triggered
        """
        if not self.config.enable_sentiment_triggers:
            return True

        # Check for significant divergence
        if market_direction != sentiment_direction and divergence_magnitude > 0.7:
            self._trip(
                TripReason.SENTIMENT_DIVERGENCE,
                {
                    "symbol": symbol,
                    "market_direction": market_direction,
                    "sentiment_direction": sentiment_direction,
                    "divergence_magnitude": divergence_magnitude,
                },
            )
            return False

        return True

    def _record_sentiment(
        self,
        symbol: str,
        score: float,
        confidence: float,
        source: str,
    ) -> None:
        """Record sentiment signal in history."""
        self._sentiment_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "score": score,
                "confidence": confidence,
                "source": source,
            }
        )

        # Trim history
        if len(self._sentiment_history) > self._max_sentiment_history:
            self._sentiment_history = self._sentiment_history[-self._max_sentiment_history :]

    def get_sentiment_stats(self) -> dict:
        """
        Get sentiment-related statistics (UPGRADE-014).

        Returns:
            Dictionary with sentiment stats
        """
        if not self._sentiment_history:
            return {
                "total_signals": 0,
                "avg_sentiment": 0.0,
                "negative_count": 0,
                "positive_count": 0,
                "consecutive_negative_by_symbol": {},
            }

        scores = [s["score"] for s in self._sentiment_history]
        return {
            "total_signals": len(self._sentiment_history),
            "avg_sentiment": sum(scores) / len(scores),
            "negative_count": sum(1 for s in scores if s < -0.2),
            "positive_count": sum(1 for s in scores if s > 0.2),
            "neutral_count": sum(1 for s in scores if -0.2 <= s <= 0.2),
            "min_sentiment": min(scores),
            "max_sentiment": max(scores),
            "consecutive_negative_by_symbol": dict(self._consecutive_negative_sentiment),
            "recent_signals": self._sentiment_history[-5:],
        }

    def reset_sentiment_tracking(self, symbol: str | None = None) -> None:
        """
        Reset sentiment tracking (UPGRADE-014).

        Args:
            symbol: Reset for specific symbol, or all if None
        """
        if symbol:
            self._consecutive_negative_sentiment.pop(symbol, None)
        else:
            self._consecutive_negative_sentiment.clear()
            self._sentiment_history.clear()

    def _trip(self, reason: TripReason, details: dict) -> None:
        """
        Trip the circuit breaker.

        Args:
            reason: Reason for tripping
            details: Additional details
        """
        # Preserve first trip reason if already tripped
        if self._state == CircuitBreakerState.OPEN and self._trip_reason is not None:
            # Already tripped - just log the additional condition but preserve original reason
            logger.warning(
                f"Circuit breaker already tripped ({self._trip_reason.value}), " f"additional condition: {reason.value}"
            )
            return

        self._state = CircuitBreakerState.OPEN
        self._trip_reason = reason
        self._trip_time = datetime.now()

        event = CircuitBreakerEvent(
            timestamp=self._trip_time,
            state=CircuitBreakerState.OPEN,
            reason=reason,
            details=details,
        )
        self._events.append(event)

        # Log the event
        logger.warning(f"CIRCUIT BREAKER TRIPPED: {reason.value} - {details}")

        # Call alert callback if configured
        if self.alert_callback:
            self.alert_callback(
                f"Trading halted: {reason.value}",
                {"reason": reason.value, "details": details, "timestamp": str(self._trip_time)},
            )

        # Write to log file
        self._write_log(event)

    def halt_all_trading(self, reason: str = "manual halt") -> None:
        """
        Manually halt all trading.

        Args:
            reason: Reason for manual halt
        """
        self._trip(TripReason.MANUAL_HALT, {"reason": reason})

    def alert_human(self, message: str) -> None:
        """
        Send alert to human operator.

        Args:
            message: Alert message
        """
        logger.critical(f"HUMAN ALERT: {message}")
        if self.alert_callback:
            self.alert_callback(message, {"type": "human_alert", "timestamp": str(datetime.now())})

    def reset(self, authorized_by: str) -> bool:
        """
        Reset the circuit breaker (requires authorization).

        Args:
            authorized_by: Name/ID of person authorizing reset

        Returns:
            True if reset successful
        """
        if not self.config.require_human_reset:
            # Auto-reset if human reset not required
            return self._do_reset(authorized_by)

        # Check cooldown period
        if self._trip_time:
            elapsed = (datetime.now() - self._trip_time).total_seconds() / 60
            if elapsed < self.config.cooldown_minutes:
                logger.warning(
                    f"Cannot reset: cooldown period not elapsed ({elapsed:.1f}/{self.config.cooldown_minutes} min)"
                )
                return False

        return self._do_reset(authorized_by)

    def _do_reset(self, authorized_by: str) -> bool:
        """Perform the actual reset."""
        self._state = CircuitBreakerState.CLOSED
        self._trip_reason = None
        self._consecutive_losses = 0

        event = CircuitBreakerEvent(
            timestamp=datetime.now(),
            state=CircuitBreakerState.CLOSED,
            resolved=True,
            resolved_by=authorized_by,
        )
        self._events.append(event)

        logger.info(f"Circuit breaker RESET by {authorized_by}")
        self._write_log(event)
        return True

    def get_status(self) -> dict:
        """
        Get current circuit breaker status.

        Returns:
            Status dictionary
        """
        return {
            "state": self._state.value,
            "can_trade": self.can_trade(),
            "trip_reason": self._trip_reason.value if self._trip_reason else None,
            "trip_time": str(self._trip_time) if self._trip_time else None,
            "consecutive_losses": self._consecutive_losses,
            "config": {
                "max_daily_loss_pct": self.config.max_daily_loss_pct,
                "max_drawdown_pct": self.config.max_drawdown_pct,
                "max_consecutive_losses": self.config.max_consecutive_losses,
                "require_human_reset": self.config.require_human_reset,
            },
        }

    def check_from_algorithm(self, algorithm) -> tuple[bool, str | None]:
        """
        Run all circuit breaker checks using QuantConnect algorithm data.

        INTEGRATION: Call this from algorithm.OnData() to monitor trading

        Example:
            def OnData(self, slice):
                if self.IsWarmingUp:
                    return

                # Check circuit breaker
                can_trade, reason = self.circuit_breaker.check_from_algorithm(self)
                if not can_trade:
                    self.Debug(f"Trading halted by circuit breaker: {reason}")
                    return

                # Continue with trading logic...

        Args:
            algorithm: QCAlgorithm instance

        Returns:
            Tuple of (can_trade, reason_if_halted)
        """
        # Set algorithm reference for logging compatibility
        if self._algorithm is None:
            self._algorithm = algorithm

        # Update peak equity
        current_equity = algorithm.Portfolio.TotalPortfolioValue
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # Check daily loss
        # Note: Assumes daily_starting_equity is tracked separately
        if hasattr(self, "_daily_starting_equity"):
            daily_pnl_pct = (current_equity - self._daily_starting_equity) / self._daily_starting_equity
        else:
            # Fallback: use Portfolio.NetProfit as approximation
            daily_pnl_pct = (
                algorithm.Portfolio.NetProfit / algorithm.Portfolio.TotalPortfolioValue
                if algorithm.Portfolio.TotalPortfolioValue > 0
                else 0
            )

        if daily_pnl_pct < -self.config.max_daily_loss_pct:
            self._trip(
                TripReason.DAILY_LOSS,
                {"daily_pnl_pct": daily_pnl_pct, "limit": self.config.max_daily_loss_pct},
            )
            algorithm.Debug(f"CIRCUIT BREAKER TRIPPED: Daily loss limit breached {daily_pnl_pct:.2%}")
            return False, f"Daily loss limit breached: {daily_pnl_pct:.2%}"

        # Check drawdown
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown > self.config.max_drawdown_pct:
                self._trip(
                    TripReason.MAX_DRAWDOWN,
                    {"drawdown": drawdown, "limit": self.config.max_drawdown_pct},
                )
                algorithm.Debug(f"CIRCUIT BREAKER TRIPPED: Max drawdown exceeded {drawdown:.2%}")
                return False, f"Max drawdown exceeded: {drawdown:.2%}"

        # Check consecutive losses
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self._trip(
                TripReason.CONSECUTIVE_LOSSES,
                {
                    "consecutive_losses": self._consecutive_losses,
                    "limit": self.config.max_consecutive_losses,
                },
            )
            algorithm.Debug(f"CIRCUIT BREAKER TRIPPED: {self._consecutive_losses} consecutive losses")
            return False, f"Consecutive losses exceeded: {self._consecutive_losses}"

        return True, None

    def record_trade_from_order_event(self, algorithm, order_event) -> None:
        """
        Record trade results from QuantConnect OnOrderEvent.

        INTEGRATION: Call this from algorithm.OnOrderEvent()

        Example:
            def OnOrderEvent(self, order_event):
                from AlgorithmImports import OrderStatus

                if order_event.Status == OrderStatus.Filled:
                    self.circuit_breaker.record_trade_from_order_event(self, order_event)

        Args:
            algorithm: QCAlgorithm instance
            order_event: OrderEvent from OnOrderEvent
        """
        try:
            from AlgorithmImports import OrderStatus

            if order_event.Status != OrderStatus.Filled:
                return

            # Determine if trade was profitable
            # This is a simplified heuristic - you may need to track entry prices
            # Note: Using GetOrderTicket (official QuantConnect API)
            ticket = algorithm.Transactions.GetOrderTicket(order_event.OrderId)
            if ticket:
                # Check if this completes a round-trip trade
                # For now, record based on unrealized P&L
                symbol = order_event.Symbol
                if symbol in algorithm.Portfolio and algorithm.Portfolio[symbol].Invested:
                    unrealized_pnl = algorithm.Portfolio[symbol].UnrealizedProfit
                    is_winner = unrealized_pnl > 0

                    self.record_trade_result(is_winner)

                    if not is_winner:
                        algorithm.Debug(f"Circuit breaker: Consecutive losses = {self._consecutive_losses}")

        except ImportError:
            algorithm.Debug("OrderStatus not available - need AlgorithmImports")
        except Exception as e:
            algorithm.Debug(f"Error recording trade in circuit breaker: {e}")

    def reset_daily_stats_qc(self, algorithm) -> None:
        """
        Reset daily statistics at start of new trading day.

        INTEGRATION: Call this at market open

        Example:
            def Initialize(self):
                # Schedule daily reset
                self.Schedule.On(
                    self.DateRules.EveryDay("SPY"),
                    self.TimeRules.AfterMarketOpen("SPY", 1),
                    lambda: self.circuit_breaker.reset_daily_stats_qc(self)
                )

        Args:
            algorithm: QCAlgorithm instance
        """
        # Set algorithm reference for logging compatibility
        if self._algorithm is None:
            self._algorithm = algorithm

        self._daily_starting_equity = algorithm.Portfolio.TotalPortfolioValue
        self._consecutive_losses = 0  # Optional: reset consecutive losses daily
        algorithm.Debug(f"Circuit breaker daily reset - Starting equity: ${self._daily_starting_equity:,.2f}")

    def reset_qc(self, algorithm, authorized_by: str) -> bool:
        """
        Reset circuit breaker from QuantConnect algorithm.

        INTEGRATION: Call this to reset after manual review

        Example:
            # In algorithm or external control
            if self.circuit_breaker.is_open:
                # After human review
                success = self.circuit_breaker.reset_qc(self, "trader@example.com")
                if success:
                    self.Debug("Circuit breaker reset - trading resumed")

        Args:
            algorithm: QCAlgorithm instance
            authorized_by: Email/ID of person authorizing reset

        Returns:
            True if reset successful
        """
        result = self.reset(authorized_by)
        if result:
            algorithm.Debug(f"Circuit breaker RESET by {authorized_by} - Trading resumed")
        else:
            algorithm.Debug("Circuit breaker reset FAILED - cooldown period not elapsed")
        return result

    def _write_log(self, event: CircuitBreakerEvent) -> None:
        """
        Write event to log.

        QUANTCONNECT COMPATIBILITY:
        - In QuantConnect cloud, file I/O is not allowed
        - Instead, we use algorithm.Debug() or ObjectStore
        - This method safely handles both environments
        """
        try:
            log_entry = {
                "timestamp": str(event.timestamp),
                "state": event.state.value,
                "reason": event.reason.value if event.reason else None,
                "details": event.details,
                "resolved": event.resolved,
                "resolved_by": event.resolved_by,
            }

            # If running in QuantConnect, use algorithm.Debug() instead of file I/O
            if hasattr(self, "_algorithm") and self._algorithm is not None:
                # Use QuantConnect logging
                log_msg = f"CircuitBreaker: {event.state.value}"
                if event.reason:
                    log_msg += f" - {event.reason.value}"
                if event.details:
                    log_msg += f" - {event.details}"
                self._algorithm.Debug(log_msg)

                # Optional: Use ObjectStore for persistent storage
                # Note: ObjectStore has size limits, so we keep only recent events
                try:
                    store_key = "circuit_breaker_events"
                    stored_events = (
                        self._algorithm.ObjectStore.Read(store_key)
                        if self._algorithm.ObjectStore.ContainsKey(store_key)
                        else "[]"
                    )
                    logs = json.loads(stored_events)
                    logs.append(log_entry)

                    # Keep only last 100 events to manage storage
                    if len(logs) > 100:
                        logs = logs[-100:]

                    self._algorithm.ObjectStore.Save(store_key, json.dumps(logs))
                except Exception as store_error:
                    # If ObjectStore fails, just use Debug logging
                    logger.debug(f"ObjectStore unavailable: {store_error}")
            else:
                # Running locally - use file I/O
                logs = []
                if self.log_file.exists():
                    with open(self.log_file) as f:
                        logs = json.load(f)

                logs.append(log_entry)

                with open(self.log_file, "w") as f:
                    json.dump(logs, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to write circuit breaker log: {e}")


# Convenience function for creating a configured circuit breaker
def create_circuit_breaker(
    max_daily_loss: float = 0.03,
    max_drawdown: float = 0.10,
    max_consecutive_losses: int = 5,
    require_human_reset: bool = True,
) -> TradingCircuitBreaker:
    """
    Create a configured circuit breaker.

    Args:
        max_daily_loss: Maximum daily loss percentage (e.g., 0.03 for 3%)
        max_drawdown: Maximum drawdown percentage (e.g., 0.10 for 10%)
        max_consecutive_losses: Maximum consecutive losing trades
        require_human_reset: Whether human intervention is required to reset

    Returns:
        Configured TradingCircuitBreaker instance
    """
    config = CircuitBreakerConfig(
        max_daily_loss_pct=max_daily_loss,
        max_drawdown_pct=max_drawdown,
        max_consecutive_losses=max_consecutive_losses,
        require_human_reset=require_human_reset,
    )
    return TradingCircuitBreaker(config=config)
