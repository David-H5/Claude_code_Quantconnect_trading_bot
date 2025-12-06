"""
Execution Quality Metrics

Aggregates and reports on order execution quality metrics.
Provides a unified view of fill rate, slippage, latency, and cancel rate.

Refactored: Phase 3 - Consolidated Metrics Infrastructure

Location: observability/metrics/collectors/execution.py
Old location: execution/execution_quality_metrics.py (re-exports for compatibility)

Features:
- Unified execution quality dashboard
- Fill rate tracking
- Latency analysis
- Cancel/reject rate monitoring
- Integration with SlippageMonitor
- Report generation

QuantConnect Compatible: Yes
- Non-blocking design
- Memory-efficient
- Configurable thresholds
"""

import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class OrderStatus(Enum):
    """Order execution status."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderRecord:
    """Record of an order for metrics tracking."""

    order_id: str
    symbol: str
    quantity: int
    filled_quantity: int
    side: str
    order_type: str
    status: OrderStatus
    submit_time: datetime
    fill_time: datetime | None
    latency_ms: float
    expected_price: float
    fill_price: float | None
    cancel_reason: str | None = None
    reject_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fill_rate(self) -> float:
        """Calculate fill rate for this order."""
        if self.quantity <= 0:
            return 0.0
        return self.filled_quantity / self.quantity

    @property
    def is_filled(self) -> bool:
        """Check if order was fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_cancelled(self) -> bool:
        """Check if order was cancelled."""
        return self.status == OrderStatus.CANCELLED

    @property
    def is_rejected(self) -> bool:
        """Check if order was rejected."""
        return self.status == OrderStatus.REJECTED


@dataclass
class ExecutionDashboard:
    """Aggregated execution quality metrics for dashboard display."""

    # Time period
    period_start: datetime
    period_end: datetime

    # Volume metrics
    total_orders: int
    total_volume: int
    filled_volume: int

    # Fill metrics
    fill_rate_pct: float
    full_fill_rate_pct: float
    partial_fill_rate_pct: float

    # Slippage metrics
    avg_slippage_bps: float
    median_slippage_bps: float
    max_slippage_bps: float
    total_slippage_cost: float

    # Latency metrics
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Cancel/Reject metrics
    cancel_rate_pct: float
    reject_rate_pct: float
    top_cancel_reasons: list[tuple[str, int]]
    top_reject_reasons: list[tuple[str, int]]

    # Quality score
    quality_score: float  # 0-100 composite score

    # Thresholds status
    fill_rate_status: str  # "good", "warning", "critical"
    slippage_status: str
    latency_status: str
    cancel_rate_status: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_orders": self.total_orders,
            "total_volume": self.total_volume,
            "filled_volume": self.filled_volume,
            "fill_rate_pct": self.fill_rate_pct,
            "full_fill_rate_pct": self.full_fill_rate_pct,
            "partial_fill_rate_pct": self.partial_fill_rate_pct,
            "avg_slippage_bps": self.avg_slippage_bps,
            "median_slippage_bps": self.median_slippage_bps,
            "max_slippage_bps": self.max_slippage_bps,
            "total_slippage_cost": self.total_slippage_cost,
            "avg_latency_ms": self.avg_latency_ms,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "cancel_rate_pct": self.cancel_rate_pct,
            "reject_rate_pct": self.reject_rate_pct,
            "quality_score": self.quality_score,
            "fill_rate_status": self.fill_rate_status,
            "slippage_status": self.slippage_status,
            "latency_status": self.latency_status,
            "cancel_rate_status": self.cancel_rate_status,
        }


@dataclass
class QualityThresholds:
    """Thresholds for execution quality metrics."""

    # Fill rate thresholds
    fill_rate_good: float = 80.0
    fill_rate_warning: float = 50.0

    # Slippage thresholds (basis points)
    slippage_good: float = 5.0
    slippage_warning: float = 15.0

    # Latency thresholds (milliseconds)
    latency_good: float = 100.0
    latency_warning: float = 500.0

    # Cancel rate thresholds
    cancel_rate_good: float = 10.0
    cancel_rate_warning: float = 30.0


class ExecutionQualityTracker:
    """
    Tracks and reports on execution quality metrics.

    Provides a unified view of fill rate, slippage, latency, and cancel rate.

    Usage:
        tracker = ExecutionQualityTracker()

        # Record orders
        tracker.record_order(
            order_id="ORD001",
            symbol="SPY",
            quantity=100,
            filled_quantity=100,
            side="buy",
            order_type="limit",
            status=OrderStatus.FILLED,
            submit_time=datetime.utcnow() - timedelta(milliseconds=50),
            fill_time=datetime.utcnow(),
            latency_ms=50.0,
            expected_price=450.00,
            fill_price=450.05,
        )

        # Get dashboard
        dashboard = tracker.get_dashboard()
        print(f"Fill Rate: {dashboard.fill_rate_pct:.1f}%")
        print(f"Quality Score: {dashboard.quality_score:.1f}")
    """

    def __init__(
        self,
        thresholds: QualityThresholds | None = None,
        max_history: int = 10000,
        slippage_monitor: Any | None = None,
    ):
        """
        Initialize execution quality tracker.

        Args:
            thresholds: Quality thresholds (defaults to standard)
            max_history: Maximum orders to keep in history
            slippage_monitor: Optional SlippageMonitor for slippage data
        """
        self.thresholds = thresholds or QualityThresholds()
        self.max_history = max_history
        self.slippage_monitor = slippage_monitor

        # Order history
        self.orders: deque = deque(maxlen=max_history)

        # Counters for quick stats
        self._total_orders = 0
        self._filled_orders = 0
        self._cancelled_orders = 0
        self._rejected_orders = 0

    def record_order(
        self,
        order_id: str,
        symbol: str,
        quantity: int,
        filled_quantity: int,
        side: str,
        order_type: str,
        status: OrderStatus,
        submit_time: datetime,
        fill_time: datetime | None,
        latency_ms: float,
        expected_price: float,
        fill_price: float | None = None,
        cancel_reason: str | None = None,
        reject_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OrderRecord:
        """
        Record an order for metrics tracking.

        Args:
            order_id: Unique order identifier
            symbol: Security symbol
            quantity: Order quantity
            filled_quantity: Quantity filled
            side: Order side (buy/sell)
            order_type: Order type (market/limit/etc.)
            status: Order status
            submit_time: Order submission time
            fill_time: Fill time (if filled)
            latency_ms: Order latency in milliseconds
            expected_price: Expected fill price
            fill_price: Actual fill price
            cancel_reason: Reason for cancellation
            reject_reason: Reason for rejection
            metadata: Additional metadata

        Returns:
            OrderRecord
        """
        record = OrderRecord(
            order_id=order_id,
            symbol=symbol,
            quantity=quantity,
            filled_quantity=filled_quantity,
            side=side,
            order_type=order_type,
            status=status,
            submit_time=submit_time,
            fill_time=fill_time,
            latency_ms=latency_ms,
            expected_price=expected_price,
            fill_price=fill_price,
            cancel_reason=cancel_reason,
            reject_reason=reject_reason,
            metadata=metadata or {},
        )

        self.orders.append(record)

        # Update counters
        self._total_orders += 1
        if status == OrderStatus.FILLED:
            self._filled_orders += 1
        elif status == OrderStatus.CANCELLED:
            self._cancelled_orders += 1
        elif status == OrderStatus.REJECTED:
            self._rejected_orders += 1

        # Also record to slippage monitor if available
        if self.slippage_monitor and fill_price is not None and filled_quantity > 0:
            self.slippage_monitor.record_fill(
                order_id=order_id,
                symbol=symbol,
                expected_price=expected_price,
                actual_price=fill_price,
                quantity=filled_quantity,
                side=side,
                order_type=order_type,
                timestamp=fill_time,
            )

        return record

    def get_dashboard(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> ExecutionDashboard:
        """
        Get execution quality dashboard.

        Args:
            start_time: Filter start time
            end_time: Filter end time

        Returns:
            ExecutionDashboard with aggregated metrics
        """
        orders = list(self.orders)

        # Apply time filters
        if start_time:
            orders = [o for o in orders if o.submit_time >= start_time]
        if end_time:
            orders = [o for o in orders if o.submit_time <= end_time]

        now = datetime.now(timezone.utc)
        if not orders:
            return self._empty_dashboard(start_time or now, end_time or now)

        # Calculate metrics
        total_orders = len(orders)
        total_volume = sum(o.quantity for o in orders)
        filled_volume = sum(o.filled_quantity for o in orders)

        # Fill rates
        filled_orders = [o for o in orders if o.status == OrderStatus.FILLED]
        partial_orders = [o for o in orders if o.status == OrderStatus.PARTIALLY_FILLED]
        cancelled_orders = [o for o in orders if o.status == OrderStatus.CANCELLED]
        rejected_orders = [o for o in orders if o.status == OrderStatus.REJECTED]

        fill_rate_pct = (len(filled_orders) + len(partial_orders)) / total_orders * 100 if total_orders > 0 else 0.0
        full_fill_rate_pct = len(filled_orders) / total_orders * 100 if total_orders > 0 else 0.0
        partial_fill_rate_pct = len(partial_orders) / total_orders * 100 if total_orders > 0 else 0.0
        cancel_rate_pct = len(cancelled_orders) / total_orders * 100 if total_orders > 0 else 0.0
        reject_rate_pct = len(rejected_orders) / total_orders * 100 if total_orders > 0 else 0.0

        # Slippage metrics
        slippage_data = self._calculate_slippage_metrics(orders)

        # Latency metrics
        latency_data = self._calculate_latency_metrics(orders)

        # Cancel/reject reasons
        top_cancel_reasons = self._get_top_reasons(cancelled_orders, "cancel_reason")
        top_reject_reasons = self._get_top_reasons(rejected_orders, "reject_reason")

        # Quality score
        quality_score = self._calculate_quality_score(
            fill_rate_pct,
            slippage_data["avg"],
            latency_data["avg"],
            cancel_rate_pct,
        )

        # Status indicators
        fill_rate_status = self._get_status(
            fill_rate_pct,
            self.thresholds.fill_rate_good,
            self.thresholds.fill_rate_warning,
            higher_is_better=True,
        )
        slippage_status = self._get_status(
            slippage_data["avg"],
            self.thresholds.slippage_good,
            self.thresholds.slippage_warning,
            higher_is_better=False,
        )
        latency_status = self._get_status(
            latency_data["avg"],
            self.thresholds.latency_good,
            self.thresholds.latency_warning,
            higher_is_better=False,
        )
        cancel_rate_status = self._get_status(
            cancel_rate_pct,
            self.thresholds.cancel_rate_good,
            self.thresholds.cancel_rate_warning,
            higher_is_better=False,
        )

        return ExecutionDashboard(
            period_start=min(o.submit_time for o in orders),
            period_end=max(o.submit_time for o in orders),
            total_orders=total_orders,
            total_volume=total_volume,
            filled_volume=filled_volume,
            fill_rate_pct=fill_rate_pct,
            full_fill_rate_pct=full_fill_rate_pct,
            partial_fill_rate_pct=partial_fill_rate_pct,
            avg_slippage_bps=slippage_data["avg"],
            median_slippage_bps=slippage_data["median"],
            max_slippage_bps=slippage_data["max"],
            total_slippage_cost=slippage_data["cost"],
            avg_latency_ms=latency_data["avg"],
            median_latency_ms=latency_data["median"],
            p95_latency_ms=latency_data["p95"],
            p99_latency_ms=latency_data["p99"],
            cancel_rate_pct=cancel_rate_pct,
            reject_rate_pct=reject_rate_pct,
            top_cancel_reasons=top_cancel_reasons,
            top_reject_reasons=top_reject_reasons,
            quality_score=quality_score,
            fill_rate_status=fill_rate_status,
            slippage_status=slippage_status,
            latency_status=latency_status,
            cancel_rate_status=cancel_rate_status,
        )

    def _calculate_slippage_metrics(self, orders: list[OrderRecord]) -> dict[str, float]:
        """Calculate slippage metrics from orders."""
        # Get filled orders with prices
        filled_with_prices = [o for o in orders if o.fill_price is not None and o.expected_price > 0]

        if not filled_with_prices:
            return {"avg": 0.0, "median": 0.0, "max": 0.0, "cost": 0.0}

        slippages = []
        total_cost = 0.0

        for order in filled_with_prices:
            diff = order.fill_price - order.expected_price
            slippage_pct = diff / order.expected_price
            slippage_bps = abs(slippage_pct * 10000)
            slippages.append(slippage_bps)

            # Calculate cost (adverse slippage)
            if (order.side.lower() == "buy" and diff > 0) or (order.side.lower() == "sell" and diff < 0):
                total_cost += abs(diff) * order.filled_quantity

        return {
            "avg": statistics.mean(slippages),
            "median": statistics.median(slippages),
            "max": max(slippages),
            "cost": total_cost,
        }

    def _calculate_latency_metrics(self, orders: list[OrderRecord]) -> dict[str, float]:
        """Calculate latency metrics from orders."""
        latencies = [o.latency_ms for o in orders if o.latency_ms > 0]

        if not latencies:
            return {"avg": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return {
            "avg": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
        }

    def _get_top_reasons(
        self,
        orders: list[OrderRecord],
        reason_field: str,
        top_n: int = 5,
    ) -> list[tuple[str, int]]:
        """Get top N reasons from orders."""
        reason_counts: dict[str, int] = {}

        for order in orders:
            reason = getattr(order, reason_field, None)
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        sorted_reasons = sorted(
            reason_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_reasons[:top_n]

    def _calculate_quality_score(
        self,
        fill_rate: float,
        avg_slippage: float,
        avg_latency: float,
        cancel_rate: float,
    ) -> float:
        """
        Calculate composite quality score (0-100).

        Weights:
        - Fill rate: 40%
        - Slippage: 25%
        - Latency: 20%
        - Cancel rate: 15%
        """
        # Normalize each metric to 0-100
        # Fill rate: already 0-100
        fill_score = min(fill_rate, 100.0)

        # Slippage: 0 bps = 100, 20+ bps = 0
        slippage_score = max(0, 100 - (avg_slippage * 5))

        # Latency: <50ms = 100, >1000ms = 0
        latency_score = max(0, 100 - (avg_latency / 10))

        # Cancel rate: 0% = 100, 50%+ = 0
        cancel_score = max(0, 100 - (cancel_rate * 2))

        # Weighted average
        quality_score = fill_score * 0.40 + slippage_score * 0.25 + latency_score * 0.20 + cancel_score * 0.15

        return round(quality_score, 1)

    def _get_status(
        self,
        value: float,
        good_threshold: float,
        warning_threshold: float,
        higher_is_better: bool,
    ) -> str:
        """Determine status based on thresholds."""
        if higher_is_better:
            if value >= good_threshold:
                return "good"
            elif value >= warning_threshold:
                return "warning"
            else:
                return "critical"
        else:
            if value <= good_threshold:
                return "good"
            elif value <= warning_threshold:
                return "warning"
            else:
                return "critical"

    def _empty_dashboard(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> ExecutionDashboard:
        """Create empty dashboard for no data."""
        return ExecutionDashboard(
            period_start=start_time,
            period_end=end_time,
            total_orders=0,
            total_volume=0,
            filled_volume=0,
            fill_rate_pct=0.0,
            full_fill_rate_pct=0.0,
            partial_fill_rate_pct=0.0,
            avg_slippage_bps=0.0,
            median_slippage_bps=0.0,
            max_slippage_bps=0.0,
            total_slippage_cost=0.0,
            avg_latency_ms=0.0,
            median_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            cancel_rate_pct=0.0,
            reject_rate_pct=0.0,
            top_cancel_reasons=[],
            top_reject_reasons=[],
            quality_score=0.0,
            fill_rate_status="good",
            slippage_status="good",
            latency_status="good",
            cancel_rate_status="good",
        )

    def clear(self) -> None:
        """Clear all order history."""
        self.orders.clear()
        self._total_orders = 0
        self._filled_orders = 0
        self._cancelled_orders = 0
        self._rejected_orders = 0


def create_execution_tracker(
    thresholds: QualityThresholds | None = None,
    **kwargs,
) -> ExecutionQualityTracker:
    """
    Factory function to create an execution quality tracker.

    Args:
        thresholds: Quality thresholds
        **kwargs: Additional arguments

    Returns:
        Configured ExecutionQualityTracker
    """
    return ExecutionQualityTracker(thresholds=thresholds, **kwargs)


def generate_execution_report(
    tracker: ExecutionQualityTracker,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> str:
    """
    Generate an execution quality report.

    Args:
        tracker: ExecutionQualityTracker instance
        start_time: Report start time
        end_time: Report end time

    Returns:
        Formatted report string
    """
    dashboard = tracker.get_dashboard(start_time, end_time)

    status_emoji = {
        "good": "✅",
        "warning": "⚠️",
        "critical": "❌",
    }

    lines = [
        "=" * 60,
        "EXECUTION QUALITY REPORT",
        "=" * 60,
        "",
        f"Period: {dashboard.period_start.strftime('%Y-%m-%d %H:%M')} to "
        f"{dashboard.period_end.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"QUALITY SCORE: {dashboard.quality_score:.1f}/100",
        "",
        "FILL METRICS",
        "-" * 40,
        f"Total Orders:     {dashboard.total_orders:,}",
        f"Total Volume:     {dashboard.total_volume:,}",
        f"Filled Volume:    {dashboard.filled_volume:,}",
        f"Fill Rate:        {dashboard.fill_rate_pct:.1f}% {status_emoji[dashboard.fill_rate_status]}",
        f"Full Fill Rate:   {dashboard.full_fill_rate_pct:.1f}%",
        f"Partial Fill:     {dashboard.partial_fill_rate_pct:.1f}%",
        "",
        "SLIPPAGE METRICS",
        "-" * 40,
        f"Average:          {dashboard.avg_slippage_bps:.2f} bps {status_emoji[dashboard.slippage_status]}",
        f"Median:           {dashboard.median_slippage_bps:.2f} bps",
        f"Maximum:          {dashboard.max_slippage_bps:.2f} bps",
        f"Total Cost:       ${dashboard.total_slippage_cost:,.2f}",
        "",
        "LATENCY METRICS",
        "-" * 40,
        f"Average:          {dashboard.avg_latency_ms:.1f}ms {status_emoji[dashboard.latency_status]}",
        f"Median:           {dashboard.median_latency_ms:.1f}ms",
        f"P95:              {dashboard.p95_latency_ms:.1f}ms",
        f"P99:              {dashboard.p99_latency_ms:.1f}ms",
        "",
        "CANCEL/REJECT METRICS",
        "-" * 40,
        f"Cancel Rate:      {dashboard.cancel_rate_pct:.1f}% {status_emoji[dashboard.cancel_rate_status]}",
        f"Reject Rate:      {dashboard.reject_rate_pct:.1f}%",
    ]

    if dashboard.top_cancel_reasons:
        lines.append("\nTop Cancel Reasons:")
        for reason, count in dashboard.top_cancel_reasons[:3]:
            lines.append(f"  - {reason}: {count}")

    if dashboard.top_reject_reasons:
        lines.append("\nTop Reject Reasons:")
        for reason, count in dashboard.top_reject_reasons[:3]:
            lines.append(f"  - {reason}: {count}")

    lines.extend(["", "=" * 60])

    return "\n".join(lines)
