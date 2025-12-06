"""
Anti-Manipulation Detection Module

UPGRADE-015 Phase 11: Compliance and Audit Logging

Detects patterns that may indicate market manipulation:
- Spoofing detection
- Layering detection
- Wash trading detection
- Momentum ignition detection
- Quote stuffing detection

SEC/FINRA Compliance:
- Rule 10b-5: Prohibits fraud and manipulation
- Dodd-Frank: Enhanced anti-manipulation provisions
- FINRA Rule 5210: Publication of quotations
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class ManipulationType(Enum):
    """Types of market manipulation."""

    SPOOFING = "spoofing"  # Orders intended to be cancelled
    LAYERING = "layering"  # Multiple orders at different prices
    WASH_TRADING = "wash_trading"  # Trading with oneself
    MOMENTUM_IGNITION = "momentum_ignition"  # Aggressive orders to trigger movement
    QUOTE_STUFFING = "quote_stuffing"  # Excessive quote updates
    MARKING_CLOSE = "marking_close"  # Trading near close to affect settlement
    FRONT_RUNNING = "front_running"  # Trading ahead of client orders
    PAINTING_TAPE = "painting_tape"  # Creating appearance of activity


class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OrderEvent:
    """Order event for manipulation detection."""

    timestamp: datetime
    order_id: str
    symbol: str
    side: str  # buy, sell
    quantity: int
    price: float
    event_type: str  # submitted, cancelled, modified, filled
    time_in_force: str = "DAY"
    account_id: str = ""


@dataclass
class ManipulationAlert:
    """Alert for detected manipulation pattern."""

    alert_id: str
    timestamp: datetime
    manipulation_type: ManipulationType
    severity: AlertSeverity
    symbol: str
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)
    related_orders: list[str] = field(default_factory=list)
    confidence: float = 0.0  # 0-1 confidence score
    recommended_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "manipulation_type": self.manipulation_type.value,
            "severity": self.severity.value,
            "symbol": self.symbol,
            "description": self.description,
            "evidence": self.evidence,
            "related_orders": self.related_orders,
            "confidence": self.confidence,
            "recommended_action": self.recommended_action,
        }


@dataclass
class DetectionConfig:
    """Configuration for manipulation detection."""

    # Spoofing detection
    spoofing_cancel_threshold: float = 0.90  # 90% cancel rate
    spoofing_time_window_seconds: int = 60
    spoofing_min_orders: int = 5

    # Layering detection
    layering_min_levels: int = 3
    layering_price_increment_pct: float = 0.001  # 0.1%
    layering_time_window_seconds: int = 30

    # Wash trading detection
    wash_trade_time_window_seconds: int = 300  # 5 minutes
    wash_trade_price_tolerance_pct: float = 0.01  # 1%

    # Momentum ignition detection
    momentum_volume_spike_ratio: float = 3.0  # 3x average
    momentum_price_move_pct: float = 0.02  # 2%
    momentum_time_window_seconds: int = 60

    # Quote stuffing detection
    quote_stuffing_threshold: int = 100  # quotes per second
    quote_stuffing_cancel_ratio: float = 0.95


class AntiManipulationMonitor:
    """Monitor for detecting market manipulation patterns."""

    def __init__(
        self,
        config: DetectionConfig | None = None,
    ):
        """
        Initialize anti-manipulation monitor.

        Args:
            config: Detection configuration
        """
        self.config = config or DetectionConfig()

        # Order history by symbol
        self._orders: dict[str, list[OrderEvent]] = {}

        # Alerts
        self._alerts: list[ManipulationAlert] = []
        self._alert_counter = 0

        # Statistics
        self._orders_processed = 0
        self._alerts_generated = 0

    # ==========================================================================
    # Order Processing
    # ==========================================================================

    def process_order_event(self, event: OrderEvent) -> list[ManipulationAlert]:
        """
        Process an order event and check for manipulation patterns.

        Args:
            event: Order event to process

        Returns:
            List of any alerts generated
        """
        # Store event
        if event.symbol not in self._orders:
            self._orders[event.symbol] = []
        self._orders[event.symbol].append(event)

        self._orders_processed += 1

        # Clean old events (keep last hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self._orders[event.symbol] = [e for e in self._orders[event.symbol] if e.timestamp > cutoff]

        # Run detection checks
        alerts = []

        # Check each manipulation type
        spoofing_alert = self._detect_spoofing(event.symbol)
        if spoofing_alert:
            alerts.append(spoofing_alert)

        layering_alert = self._detect_layering(event.symbol)
        if layering_alert:
            alerts.append(layering_alert)

        wash_alert = self._detect_wash_trading(event.symbol)
        if wash_alert:
            alerts.append(wash_alert)

        momentum_alert = self._detect_momentum_ignition(event.symbol)
        if momentum_alert:
            alerts.append(momentum_alert)

        quote_alert = self._detect_quote_stuffing(event.symbol)
        if quote_alert:
            alerts.append(quote_alert)

        # Store alerts
        self._alerts.extend(alerts)
        self._alerts_generated += len(alerts)

        return alerts

    # ==========================================================================
    # Detection Methods
    # ==========================================================================

    def _detect_spoofing(self, symbol: str) -> ManipulationAlert | None:
        """
        Detect spoofing pattern.

        Spoofing: Placing orders with intent to cancel before execution
        to create false impression of demand/supply.
        """
        config = self.config
        events = self._orders.get(symbol, [])

        if len(events) < config.spoofing_min_orders:
            return None

        # Look at recent window
        cutoff = datetime.utcnow() - timedelta(seconds=config.spoofing_time_window_seconds)
        recent = [e for e in events if e.timestamp > cutoff]

        if len(recent) < config.spoofing_min_orders:
            return None

        # Count submissions and cancellations
        submitted = [e for e in recent if e.event_type == "submitted"]
        cancelled = [e for e in recent if e.event_type == "cancelled"]
        filled = [e for e in recent if e.event_type == "filled"]

        if not submitted:
            return None

        cancel_rate = len(cancelled) / len(submitted) if submitted else 0
        fill_rate = len(filled) / len(submitted) if submitted else 0

        # Spoofing indicators
        if cancel_rate >= config.spoofing_cancel_threshold and fill_rate < 0.10:
            # Check if cancellations are rapid
            if len(cancelled) >= 2:
                cancel_times = [e.timestamp for e in cancelled]
                avg_time_between = sum(
                    [(cancel_times[i + 1] - cancel_times[i]).total_seconds() for i in range(len(cancel_times) - 1)]
                ) / (len(cancel_times) - 1)

                if avg_time_between < 5:  # Less than 5 seconds average
                    confidence = min(0.9, cancel_rate * (1 - fill_rate))

                    return self._create_alert(
                        manipulation_type=ManipulationType.SPOOFING,
                        severity=AlertSeverity.HIGH,
                        symbol=symbol,
                        description=f"Potential spoofing detected: {cancel_rate:.1%} cancel rate with {fill_rate:.1%} fill rate",
                        evidence={
                            "cancel_rate": cancel_rate,
                            "fill_rate": fill_rate,
                            "orders_submitted": len(submitted),
                            "orders_cancelled": len(cancelled),
                            "avg_time_between_cancels": avg_time_between,
                        },
                        related_orders=[e.order_id for e in cancelled[:10]],
                        confidence=confidence,
                        recommended_action="Review order pattern and consider blocking rapid cancellations",
                    )

        return None

    def _detect_layering(self, symbol: str) -> ManipulationAlert | None:
        """
        Detect layering pattern.

        Layering: Multiple orders at incrementally different prices
        to create illusion of depth.
        """
        config = self.config
        events = self._orders.get(symbol, [])

        # Look at recent window
        cutoff = datetime.utcnow() - timedelta(seconds=config.layering_time_window_seconds)
        recent = [e for e in events if e.timestamp > cutoff and e.event_type == "submitted"]

        if len(recent) < config.layering_min_levels:
            return None

        # Group by side
        buy_orders = sorted([e for e in recent if e.side == "buy"], key=lambda x: x.price, reverse=True)
        sell_orders = sorted([e for e in recent if e.side == "sell"], key=lambda x: x.price)

        # Check for layering pattern on each side
        for orders, side in [(buy_orders, "buy"), (sell_orders, "sell")]:
            if len(orders) < config.layering_min_levels:
                continue

            # Check price increments
            prices = [o.price for o in orders]
            if len(prices) < 2:
                continue

            increments = []
            for i in range(1, len(prices)):
                if prices[i - 1] > 0:
                    increment = abs(prices[i] - prices[i - 1]) / prices[i - 1]
                    increments.append(increment)

            if not increments:
                continue

            avg_increment = sum(increments) / len(increments)

            # Check if increments are consistent (layering signature)
            if len(increments) >= 2:
                increment_variance = sum((i - avg_increment) ** 2 for i in increments) / len(increments)

                # Low variance in increments + multiple levels = layering
                if (
                    increment_variance < config.layering_price_increment_pct**2
                    and len(orders) >= config.layering_min_levels
                ):
                    confidence = 0.7 if len(orders) >= 5 else 0.5

                    return self._create_alert(
                        manipulation_type=ManipulationType.LAYERING,
                        severity=AlertSeverity.MEDIUM,
                        symbol=symbol,
                        description=f"Potential layering detected: {len(orders)} {side} orders at consistent increments",
                        evidence={
                            "side": side,
                            "num_levels": len(orders),
                            "avg_increment_pct": avg_increment * 100,
                            "increment_variance": increment_variance,
                            "price_range": [min(prices), max(prices)],
                        },
                        related_orders=[o.order_id for o in orders[:10]],
                        confidence=confidence,
                        recommended_action="Monitor for rapid cancellation pattern",
                    )

        return None

    def _detect_wash_trading(self, symbol: str) -> ManipulationAlert | None:
        """
        Detect wash trading pattern.

        Wash trading: Trading with oneself to create false volume.
        """
        config = self.config
        events = self._orders.get(symbol, [])

        # Look at fills
        cutoff = datetime.utcnow() - timedelta(seconds=config.wash_trade_time_window_seconds)
        fills = [e for e in events if e.timestamp > cutoff and e.event_type == "filled"]

        if len(fills) < 2:
            return None

        # Look for matching buy/sell pairs at similar prices
        buys = [e for e in fills if e.side == "buy"]
        sells = [e for e in fills if e.side == "sell"]

        matches = []
        for buy in buys:
            for sell in sells:
                # Same account
                if buy.account_id and buy.account_id == sell.account_id:
                    # Similar price
                    price_diff = abs(buy.price - sell.price) / buy.price if buy.price > 0 else 1
                    if price_diff <= config.wash_trade_price_tolerance_pct:
                        # Close in time
                        time_diff = abs((buy.timestamp - sell.timestamp).total_seconds())
                        if time_diff <= config.wash_trade_time_window_seconds:
                            matches.append((buy, sell))

        if matches:
            total_wash_volume = sum(min(b.quantity, s.quantity) for b, s in matches)
            total_volume = sum(e.quantity for e in fills)
            wash_ratio = total_wash_volume / total_volume if total_volume > 0 else 0

            if wash_ratio > 0.3:  # 30% wash trading threshold
                confidence = min(0.9, wash_ratio)

                return self._create_alert(
                    manipulation_type=ManipulationType.WASH_TRADING,
                    severity=AlertSeverity.CRITICAL,
                    symbol=symbol,
                    description=f"Potential wash trading detected: {wash_ratio:.1%} of volume",
                    evidence={
                        "wash_volume": total_wash_volume,
                        "total_volume": total_volume,
                        "wash_ratio": wash_ratio,
                        "match_count": len(matches),
                    },
                    related_orders=[b.order_id for b, _ in matches] + [s.order_id for _, s in matches],
                    confidence=confidence,
                    recommended_action="Investigate accounts involved and report to compliance",
                )

        return None

    def _detect_momentum_ignition(self, symbol: str) -> ManipulationAlert | None:
        """
        Detect momentum ignition pattern.

        Momentum ignition: Aggressive orders to trigger price movement
        and then trading in opposite direction.
        """
        config = self.config
        events = self._orders.get(symbol, [])

        cutoff = datetime.utcnow() - timedelta(seconds=config.momentum_time_window_seconds)
        recent = [e for e in events if e.timestamp > cutoff]

        if len(recent) < 5:
            return None

        # Look for aggressive volume followed by reversal
        fills = [e for e in recent if e.event_type == "filled"]
        if len(fills) < 4:
            return None

        # Calculate volume by time segment
        segment_size = config.momentum_time_window_seconds // 4
        segments = [0, 0, 0, 0]
        segment_sides = [[], [], [], []]

        for fill in fills:
            age = (datetime.utcnow() - fill.timestamp).total_seconds()
            seg_idx = min(3, int(age / segment_size))
            segments[seg_idx] += fill.quantity
            segment_sides[seg_idx].append(fill.side)

        # Check for volume spike followed by reversal
        if segments[0] > 0 and segments[3] > 0:
            volume_ratio = segments[0] / (sum(segments[1:]) / 3) if sum(segments[1:]) > 0 else 0

            if volume_ratio >= config.momentum_volume_spike_ratio:
                # Check for direction reversal
                early_side = segment_sides[3][0] if segment_sides[3] else None
                late_side = segment_sides[0][0] if segment_sides[0] else None

                if early_side and late_side and early_side != late_side:
                    confidence = min(0.8, volume_ratio / 5)

                    return self._create_alert(
                        manipulation_type=ManipulationType.MOMENTUM_IGNITION,
                        severity=AlertSeverity.HIGH,
                        symbol=symbol,
                        description=f"Potential momentum ignition: {volume_ratio:.1f}x volume spike with direction reversal",
                        evidence={
                            "volume_ratio": volume_ratio,
                            "segment_volumes": segments,
                            "initial_side": early_side,
                            "reversal_side": late_side,
                        },
                        related_orders=[e.order_id for e in fills[:10]],
                        confidence=confidence,
                        recommended_action="Review for pattern of triggering stops and reversing",
                    )

        return None

    def _detect_quote_stuffing(self, symbol: str) -> ManipulationAlert | None:
        """
        Detect quote stuffing pattern.

        Quote stuffing: Excessive quote updates to slow down systems.
        """
        config = self.config
        events = self._orders.get(symbol, [])

        # Count events per second
        now = datetime.utcnow()
        one_second_ago = now - timedelta(seconds=1)
        events_per_second = len([e for e in events if e.timestamp > one_second_ago])

        if events_per_second >= config.quote_stuffing_threshold:
            # Calculate cancel ratio
            recent = [e for e in events if e.timestamp > one_second_ago]
            cancels = len([e for e in recent if e.event_type == "cancelled"])
            cancel_ratio = cancels / len(recent) if recent else 0

            if cancel_ratio >= config.quote_stuffing_cancel_ratio:
                confidence = min(0.9, events_per_second / config.quote_stuffing_threshold / 2)

                return self._create_alert(
                    manipulation_type=ManipulationType.QUOTE_STUFFING,
                    severity=AlertSeverity.CRITICAL,
                    symbol=symbol,
                    description=f"Potential quote stuffing: {events_per_second} events/sec with {cancel_ratio:.1%} cancels",
                    evidence={
                        "events_per_second": events_per_second,
                        "cancel_ratio": cancel_ratio,
                        "threshold": config.quote_stuffing_threshold,
                    },
                    related_orders=[e.order_id for e in recent[:20]],
                    confidence=confidence,
                    recommended_action="Implement rate limiting and investigate source",
                )

        return None

    def _create_alert(
        self,
        manipulation_type: ManipulationType,
        severity: AlertSeverity,
        symbol: str,
        description: str,
        evidence: dict[str, Any],
        related_orders: list[str],
        confidence: float,
        recommended_action: str,
    ) -> ManipulationAlert:
        """Create a new alert."""
        self._alert_counter += 1
        alert_id = f"MANIP-{datetime.utcnow().strftime('%Y%m%d')}-{self._alert_counter:05d}"

        return ManipulationAlert(
            alert_id=alert_id,
            timestamp=datetime.utcnow(),
            manipulation_type=manipulation_type,
            severity=severity,
            symbol=symbol,
            description=description,
            evidence=evidence,
            related_orders=related_orders,
            confidence=confidence,
            recommended_action=recommended_action,
        )

    # ==========================================================================
    # Retrieval and Statistics
    # ==========================================================================

    def get_alerts(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        manipulation_type: ManipulationType | None = None,
        severity: AlertSeverity | None = None,
        symbol: str | None = None,
    ) -> list[ManipulationAlert]:
        """Get filtered alerts."""
        results = []

        for alert in self._alerts:
            if start_time and alert.timestamp < start_time:
                continue
            if end_time and alert.timestamp > end_time:
                continue
            if manipulation_type and alert.manipulation_type != manipulation_type:
                continue
            if severity and alert.severity != severity:
                continue
            if symbol and alert.symbol != symbol:
                continue

            results.append(alert)

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        type_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}

        for alert in self._alerts:
            t = alert.manipulation_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

            s = alert.severity.value
            severity_counts[s] = severity_counts.get(s, 0) + 1

        return {
            "orders_processed": self._orders_processed,
            "alerts_generated": self._alerts_generated,
            "alerts_by_type": type_counts,
            "alerts_by_severity": severity_counts,
            "symbols_monitored": len(self._orders),
        }

    def clear_old_data(self, older_than_hours: int = 24) -> int:
        """Clear old order data."""
        cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)
        cleared = 0

        for symbol in list(self._orders.keys()):
            original = len(self._orders[symbol])
            self._orders[symbol] = [e for e in self._orders[symbol] if e.timestamp > cutoff]
            cleared += original - len(self._orders[symbol])

        return cleared


def create_anti_manipulation_monitor(
    spoofing_cancel_threshold: float = 0.90,
    layering_min_levels: int = 3,
    wash_trade_time_window_seconds: int = 300,
) -> AntiManipulationMonitor:
    """
    Factory function to create an anti-manipulation monitor.

    Args:
        spoofing_cancel_threshold: Cancel rate threshold for spoofing
        layering_min_levels: Minimum price levels for layering
        wash_trade_time_window_seconds: Time window for wash trading

    Returns:
        Configured AntiManipulationMonitor
    """
    config = DetectionConfig(
        spoofing_cancel_threshold=spoofing_cancel_threshold,
        layering_min_levels=layering_min_levels,
        wash_trade_time_window_seconds=wash_trade_time_window_seconds,
    )
    return AntiManipulationMonitor(config)
