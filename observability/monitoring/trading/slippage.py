"""
Slippage Monitor (CANONICAL LOCATION)

Real-time monitoring and analysis of order execution slippage.
Tracks expected vs actual fill prices to measure execution quality.

Location: observability/monitoring/trading/slippage.py
Import: from observability.monitoring.trading import SlippageMonitor

Features:
- Per-order slippage tracking
- Aggregated slippage statistics
- Alert thresholds for high slippage
- Symbol-level analysis
- Time-of-day patterns
- Persistence support

QuantConnect Compatible: Yes
- Non-blocking design
- Memory-efficient circular buffer
- Configurable alert callbacks

Part of CONSOLIDATE-001 Phase 4: Monitoring Consolidation
"""

import statistics
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SlippageDirection(Enum):
    """Direction of slippage (favorable or adverse)."""

    FAVORABLE = "favorable"  # Filled better than expected
    ADVERSE = "adverse"  # Filled worse than expected
    NEUTRAL = "neutral"  # Filled at expected price


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class FillRecord:
    """
    Record of a single order fill with slippage calculation.

    Attributes:
        order_id: Unique order identifier
        symbol: Security symbol
        expected_price: Expected fill price
        actual_price: Actual fill price
        quantity: Order quantity
        side: Order side (buy/sell)
        timestamp: Fill timestamp
        slippage_bps: Slippage in basis points
        direction: Whether slippage was favorable or adverse
        order_type: Type of order (market, limit, etc.)
        metadata: Additional order metadata
    """

    order_id: str
    symbol: str
    expected_price: float
    actual_price: float
    quantity: int
    side: str  # "buy" or "sell"
    timestamp: datetime
    slippage_bps: float
    direction: SlippageDirection
    order_type: str = "market"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "expected_price": self.expected_price,
            "actual_price": self.actual_price,
            "quantity": self.quantity,
            "side": self.side,
            "timestamp": self.timestamp.isoformat(),
            "slippage_bps": self.slippage_bps,
            "direction": self.direction.value,
            "order_type": self.order_type,
            "metadata": self.metadata,
        }


@dataclass
class SlippageAlert:
    """Alert generated when slippage exceeds thresholds."""

    level: AlertLevel
    message: str
    fill_record: FillRecord
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "order_id": self.fill_record.order_id,
            "symbol": self.fill_record.symbol,
            "slippage_bps": self.fill_record.slippage_bps,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SymbolSlippageStats:
    """Aggregated slippage statistics for a symbol."""

    symbol: str
    fill_count: int
    total_volume: int
    avg_slippage_bps: float
    median_slippage_bps: float
    max_slippage_bps: float
    min_slippage_bps: float
    std_dev_bps: float
    favorable_count: int
    adverse_count: int
    total_slippage_value: float  # Dollar value of slippage


@dataclass
class ExecutionQualityMetrics:
    """Overall execution quality metrics."""

    total_fills: int
    total_volume: int
    avg_slippage_bps: float
    median_slippage_bps: float
    max_slippage_bps: float
    min_slippage_bps: float
    std_dev_bps: float
    favorable_pct: float
    adverse_pct: float
    neutral_pct: float
    total_slippage_value: float
    alerts_count: int
    by_symbol: dict[str, SymbolSlippageStats]
    period_start: datetime
    period_end: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_fills": self.total_fills,
            "total_volume": self.total_volume,
            "avg_slippage_bps": self.avg_slippage_bps,
            "median_slippage_bps": self.median_slippage_bps,
            "max_slippage_bps": self.max_slippage_bps,
            "min_slippage_bps": self.min_slippage_bps,
            "std_dev_bps": self.std_dev_bps,
            "favorable_pct": self.favorable_pct,
            "adverse_pct": self.adverse_pct,
            "neutral_pct": self.neutral_pct,
            "total_slippage_value": self.total_slippage_value,
            "alerts_count": self.alerts_count,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
        }


class SlippageMonitor:
    """
    Real-time slippage monitoring for order execution.

    Tracks expected vs actual fill prices, calculates slippage,
    and generates alerts when thresholds are exceeded.

    Usage:
        monitor = SlippageMonitor(
            alert_threshold_bps=10.0,
            warning_threshold_bps=5.0,
        )

        # Record fills as they occur
        record = monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=450.05,
            quantity=100,
            side="buy",
        )

        # Get metrics
        metrics = monitor.get_metrics()
        print(f"Average slippage: {metrics.avg_slippage_bps:.2f} bps")
    """

    def __init__(
        self,
        alert_threshold_bps: float = 10.0,
        warning_threshold_bps: float = 5.0,
        critical_threshold_bps: float = 25.0,
        max_history: int = 10000,
        alert_callback: Callable[[SlippageAlert], None] | None = None,
    ):
        """
        Initialize slippage monitor.

        Args:
            alert_threshold_bps: Slippage threshold for warnings (basis points)
            warning_threshold_bps: Info-level threshold
            critical_threshold_bps: Critical alert threshold
            max_history: Maximum fills to keep in history
            alert_callback: Optional callback for alerts
        """
        self.alert_threshold = alert_threshold_bps
        self.warning_threshold = warning_threshold_bps
        self.critical_threshold = critical_threshold_bps
        self.max_history = max_history
        self.alert_callback = alert_callback

        # Use deque for memory-efficient circular buffer
        self.fill_history: deque = deque(maxlen=max_history)
        self.alerts: list[SlippageAlert] = []

        # Symbol-level tracking
        self._symbol_fills: dict[str, list[FillRecord]] = {}

    def record_fill(
        self,
        order_id: str,
        symbol: str,
        expected_price: float,
        actual_price: float,
        quantity: int,
        side: str,
        order_type: str = "market",
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FillRecord:
        """
        Record an order fill and calculate slippage.

        Args:
            order_id: Unique order identifier
            symbol: Security symbol
            expected_price: Expected fill price
            actual_price: Actual fill price
            quantity: Order quantity
            side: Order side ("buy" or "sell")
            order_type: Type of order
            timestamp: Fill timestamp (defaults to now)
            metadata: Additional metadata

        Returns:
            FillRecord with slippage calculation
        """
        timestamp = timestamp or datetime.now(timezone.utc)
        metadata = metadata or {}

        # Calculate slippage in basis points
        slippage_bps, direction = self._calculate_slippage_bps(expected_price, actual_price, side)

        record = FillRecord(
            order_id=order_id,
            symbol=symbol,
            expected_price=expected_price,
            actual_price=actual_price,
            quantity=quantity,
            side=side,
            timestamp=timestamp,
            slippage_bps=slippage_bps,
            direction=direction,
            order_type=order_type,
            metadata=metadata,
        )

        # Store in history
        self.fill_history.append(record)

        # Store by symbol
        if symbol not in self._symbol_fills:
            self._symbol_fills[symbol] = []
        self._symbol_fills[symbol].append(record)

        # Check alerts
        self._check_alerts(record)

        return record

    def _calculate_slippage_bps(
        self,
        expected_price: float,
        actual_price: float,
        side: str,
    ) -> tuple[float, SlippageDirection]:
        """
        Calculate slippage in basis points.

        For buys: positive slippage means paid more (adverse)
        For sells: positive slippage means received less (adverse)

        Args:
            expected_price: Expected fill price
            actual_price: Actual fill price
            side: Order side

        Returns:
            Tuple of (slippage_bps, direction)
        """
        if expected_price <= 0:
            return 0.0, SlippageDirection.NEUTRAL

        price_diff = actual_price - expected_price
        slippage_pct = price_diff / expected_price

        # Convert to basis points (1 bp = 0.01%)
        slippage_bps = slippage_pct * 10000

        # Determine direction based on side
        if side.lower() == "buy":
            # For buys, paying more is adverse
            if slippage_bps > 0.5:  # Small threshold for "neutral"
                direction = SlippageDirection.ADVERSE
            elif slippage_bps < -0.5:
                direction = SlippageDirection.FAVORABLE
            else:
                direction = SlippageDirection.NEUTRAL
        else:  # sell
            # For sells, receiving less is adverse
            if slippage_bps < -0.5:
                direction = SlippageDirection.ADVERSE
            elif slippage_bps > 0.5:
                direction = SlippageDirection.FAVORABLE
            else:
                direction = SlippageDirection.NEUTRAL

        # Return absolute value for consistent measurement
        return abs(slippage_bps), direction

    def _check_alerts(self, record: FillRecord) -> None:
        """Check if fill triggers alerts."""
        if record.direction == SlippageDirection.ADVERSE:
            abs_slippage = record.slippage_bps

            alert = None

            if abs_slippage >= self.critical_threshold:
                alert = SlippageAlert(
                    level=AlertLevel.CRITICAL,
                    message=(
                        f"CRITICAL: {record.symbol} slippage {abs_slippage:.1f}bps "
                        f"exceeds critical threshold ({self.critical_threshold}bps)"
                    ),
                    fill_record=record,
                )
            elif abs_slippage >= self.alert_threshold:
                alert = SlippageAlert(
                    level=AlertLevel.WARNING,
                    message=(
                        f"WARNING: {record.symbol} slippage {abs_slippage:.1f}bps "
                        f"exceeds alert threshold ({self.alert_threshold}bps)"
                    ),
                    fill_record=record,
                )
            elif abs_slippage >= self.warning_threshold:
                alert = SlippageAlert(
                    level=AlertLevel.INFO,
                    message=(
                        f"INFO: {record.symbol} slippage {abs_slippage:.1f}bps "
                        f"above normal ({self.warning_threshold}bps)"
                    ),
                    fill_record=record,
                )

            if alert:
                self.alerts.append(alert)
                if self.alert_callback:
                    try:
                        self.alert_callback(alert)
                    except Exception:
                        pass  # Don't let callback failures affect monitoring

    def get_metrics(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        symbol: str | None = None,
    ) -> ExecutionQualityMetrics:
        """
        Get aggregated execution quality metrics.

        Args:
            start_time: Filter start time
            end_time: Filter end time
            symbol: Filter by symbol

        Returns:
            ExecutionQualityMetrics with aggregated statistics
        """
        records = list(self.fill_history)

        # Apply filters
        if start_time:
            records = [r for r in records if r.timestamp >= start_time]
        if end_time:
            records = [r for r in records if r.timestamp <= end_time]
        if symbol:
            records = [r for r in records if r.symbol == symbol]

        if not records:
            now = datetime.now(timezone.utc)
            return ExecutionQualityMetrics(
                total_fills=0,
                total_volume=0,
                avg_slippage_bps=0.0,
                median_slippage_bps=0.0,
                max_slippage_bps=0.0,
                min_slippage_bps=0.0,
                std_dev_bps=0.0,
                favorable_pct=0.0,
                adverse_pct=0.0,
                neutral_pct=0.0,
                total_slippage_value=0.0,
                alerts_count=len(self.alerts),
                by_symbol={},
                period_start=start_time or now,
                period_end=end_time or now,
            )

        slippages = [r.slippage_bps for r in records]
        total_volume = sum(r.quantity for r in records)

        # Calculate slippage dollar value
        total_slippage_value = sum(
            r.quantity * r.expected_price * (r.slippage_bps / 10000)
            for r in records
            if r.direction == SlippageDirection.ADVERSE
        )

        # Direction counts
        favorable = len([r for r in records if r.direction == SlippageDirection.FAVORABLE])
        adverse = len([r for r in records if r.direction == SlippageDirection.ADVERSE])
        neutral = len([r for r in records if r.direction == SlippageDirection.NEUTRAL])
        total = len(records)

        # By-symbol stats
        by_symbol = {}
        symbols = set(r.symbol for r in records)
        for sym in symbols:
            sym_records = [r for r in records if r.symbol == sym]
            by_symbol[sym] = self._calculate_symbol_stats(sym, sym_records)

        return ExecutionQualityMetrics(
            total_fills=total,
            total_volume=total_volume,
            avg_slippage_bps=statistics.mean(slippages),
            median_slippage_bps=statistics.median(slippages),
            max_slippage_bps=max(slippages),
            min_slippage_bps=min(slippages),
            std_dev_bps=statistics.stdev(slippages) if len(slippages) > 1 else 0.0,
            favorable_pct=favorable / total * 100 if total > 0 else 0.0,
            adverse_pct=adverse / total * 100 if total > 0 else 0.0,
            neutral_pct=neutral / total * 100 if total > 0 else 0.0,
            total_slippage_value=total_slippage_value,
            alerts_count=len(self.alerts),
            by_symbol=by_symbol,
            period_start=min(r.timestamp for r in records),
            period_end=max(r.timestamp for r in records),
        )

    def _calculate_symbol_stats(
        self,
        symbol: str,
        records: list[FillRecord],
    ) -> SymbolSlippageStats:
        """Calculate statistics for a single symbol."""
        if not records:
            return SymbolSlippageStats(
                symbol=symbol,
                fill_count=0,
                total_volume=0,
                avg_slippage_bps=0.0,
                median_slippage_bps=0.0,
                max_slippage_bps=0.0,
                min_slippage_bps=0.0,
                std_dev_bps=0.0,
                favorable_count=0,
                adverse_count=0,
                total_slippage_value=0.0,
            )

        slippages = [r.slippage_bps for r in records]
        total_volume = sum(r.quantity for r in records)
        total_slippage_value = sum(
            r.quantity * r.expected_price * (r.slippage_bps / 10000)
            for r in records
            if r.direction == SlippageDirection.ADVERSE
        )

        return SymbolSlippageStats(
            symbol=symbol,
            fill_count=len(records),
            total_volume=total_volume,
            avg_slippage_bps=statistics.mean(slippages),
            median_slippage_bps=statistics.median(slippages),
            max_slippage_bps=max(slippages),
            min_slippage_bps=min(slippages),
            std_dev_bps=statistics.stdev(slippages) if len(slippages) > 1 else 0.0,
            favorable_count=len([r for r in records if r.direction == SlippageDirection.FAVORABLE]),
            adverse_count=len([r for r in records if r.direction == SlippageDirection.ADVERSE]),
            total_slippage_value=total_slippage_value,
        )

    def get_symbol_stats(self, symbol: str) -> SymbolSlippageStats:
        """Get statistics for a specific symbol."""
        records = self._symbol_fills.get(symbol, [])
        return self._calculate_symbol_stats(symbol, records)

    def get_recent_alerts(
        self,
        count: int = 10,
        level: AlertLevel | None = None,
    ) -> list[SlippageAlert]:
        """
        Get recent alerts.

        Args:
            count: Maximum alerts to return
            level: Filter by alert level

        Returns:
            List of recent alerts
        """
        alerts = self.alerts
        if level:
            alerts = [a for a in alerts if a.level == level]
        return alerts[-count:]

    def get_worst_fills(
        self,
        count: int = 10,
        since: datetime | None = None,
    ) -> list[FillRecord]:
        """
        Get fills with worst slippage.

        Args:
            count: Number of fills to return
            since: Only consider fills after this time

        Returns:
            List of fills sorted by slippage (worst first)
        """
        records = list(self.fill_history)

        if since:
            records = [r for r in records if r.timestamp >= since]

        # Filter to adverse fills
        adverse_fills = [r for r in records if r.direction == SlippageDirection.ADVERSE]

        # Sort by slippage (highest first)
        adverse_fills.sort(key=lambda r: r.slippage_bps, reverse=True)

        return adverse_fills[:count]

    def clear_history(self) -> None:
        """Clear all fill history."""
        self.fill_history.clear()
        self._symbol_fills.clear()
        self.alerts.clear()

    def reset_alerts(self) -> None:
        """Clear only alerts, keep fill history."""
        self.alerts.clear()


def create_slippage_monitor(
    alert_threshold_bps: float = 10.0,
    warning_threshold_bps: float = 5.0,
    **kwargs,
) -> SlippageMonitor:
    """
    Factory function to create a slippage monitor.

    Args:
        alert_threshold_bps: Alert threshold
        warning_threshold_bps: Warning threshold
        **kwargs: Additional arguments

    Returns:
        Configured SlippageMonitor
    """
    return SlippageMonitor(
        alert_threshold_bps=alert_threshold_bps,
        warning_threshold_bps=warning_threshold_bps,
        **kwargs,
    )


def generate_slippage_report(
    monitor: SlippageMonitor,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> str:
    """
    Generate a slippage analysis report.

    Args:
        monitor: SlippageMonitor instance
        start_time: Report start time
        end_time: Report end time

    Returns:
        Formatted report string
    """
    metrics = monitor.get_metrics(start_time, end_time)

    lines = [
        "=" * 60,
        "SLIPPAGE ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Period: {metrics.period_start.strftime('%Y-%m-%d %H:%M')} to "
        f"{metrics.period_end.strftime('%Y-%m-%d %H:%M')}",
        "",
        "SUMMARY",
        "-" * 40,
        f"Total Fills: {metrics.total_fills:,}",
        f"Total Volume: {metrics.total_volume:,}",
        f"Total Slippage Cost: ${metrics.total_slippage_value:,.2f}",
        "",
        "SLIPPAGE STATISTICS (basis points)",
        "-" * 40,
        f"Average:    {metrics.avg_slippage_bps:.2f} bps",
        f"Median:     {metrics.median_slippage_bps:.2f} bps",
        f"Std Dev:    {metrics.std_dev_bps:.2f} bps",
        f"Max:        {metrics.max_slippage_bps:.2f} bps",
        f"Min:        {metrics.min_slippage_bps:.2f} bps",
        "",
        "DIRECTION BREAKDOWN",
        "-" * 40,
        f"Favorable:  {metrics.favorable_pct:.1f}%",
        f"Adverse:    {metrics.adverse_pct:.1f}%",
        f"Neutral:    {metrics.neutral_pct:.1f}%",
        "",
        f"Alerts Generated: {metrics.alerts_count}",
    ]

    if metrics.by_symbol:
        lines.extend(
            [
                "",
                "BY SYMBOL",
                "-" * 40,
            ]
        )
        for symbol, stats in sorted(metrics.by_symbol.items()):
            lines.append(
                f"  {symbol}: avg={stats.avg_slippage_bps:.2f}bps, "
                f"fills={stats.fill_count}, "
                f"cost=${stats.total_slippage_value:.2f}"
            )

    lines.extend(["", "=" * 60])

    return "\n".join(lines)
