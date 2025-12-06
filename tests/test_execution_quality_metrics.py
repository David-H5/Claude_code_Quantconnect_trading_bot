"""
Tests for Execution Quality Metrics (Sprint 5)

Tests the unified execution quality tracking dashboard.
Part of UPGRADE-010 Sprint 5 - Quality & Test Coverage.
"""

from datetime import datetime, timedelta, timezone

import pytest

from execution.execution_quality_metrics import (
    ExecutionDashboard,
    ExecutionQualityTracker,
    OrderRecord,
    OrderStatus,
    QualityThresholds,
    create_execution_tracker,
    generate_execution_report,
)


class TestOrderStatus:
    """Tests for OrderStatus enum."""

    def test_all_statuses_defined(self):
        """Test all expected statuses exist."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"


class TestOrderRecord:
    """Tests for OrderRecord dataclass."""

    @pytest.fixture
    def filled_order(self):
        """Create a filled order record."""
        return OrderRecord(
            order_id="ORD001",
            symbol="SPY",
            quantity=100,
            filled_quantity=100,
            side="buy",
            order_type="limit",
            status=OrderStatus.FILLED,
            submit_time=datetime.now(timezone.utc) - timedelta(milliseconds=50),
            fill_time=datetime.now(timezone.utc),
            latency_ms=50.0,
            expected_price=450.00,
            fill_price=450.05,
        )

    @pytest.fixture
    def partial_order(self):
        """Create a partially filled order."""
        return OrderRecord(
            order_id="ORD002",
            symbol="QQQ",
            quantity=100,
            filled_quantity=50,
            side="sell",
            order_type="limit",
            status=OrderStatus.PARTIALLY_FILLED,
            submit_time=datetime.now(timezone.utc),
            fill_time=datetime.now(timezone.utc),
            latency_ms=75.0,
            expected_price=380.00,
            fill_price=379.95,
        )

    @pytest.fixture
    def cancelled_order(self):
        """Create a cancelled order."""
        return OrderRecord(
            order_id="ORD003",
            symbol="AAPL",
            quantity=50,
            filled_quantity=0,
            side="buy",
            order_type="limit",
            status=OrderStatus.CANCELLED,
            submit_time=datetime.now(timezone.utc),
            fill_time=None,
            latency_ms=100.0,
            expected_price=180.00,
            fill_price=None,
            cancel_reason="Timeout",
        )

    def test_fill_rate_full(self, filled_order):
        """Test fill rate for fully filled order."""
        assert filled_order.fill_rate == 1.0

    def test_fill_rate_partial(self, partial_order):
        """Test fill rate for partially filled order."""
        assert partial_order.fill_rate == 0.5

    def test_fill_rate_zero_quantity(self):
        """Test fill rate for zero quantity order."""
        order = OrderRecord(
            order_id="ORD",
            symbol="SPY",
            quantity=0,
            filled_quantity=0,
            side="buy",
            order_type="limit",
            status=OrderStatus.CANCELLED,
            submit_time=datetime.now(timezone.utc),
            fill_time=None,
            latency_ms=0.0,
            expected_price=450.0,
            fill_price=None,
        )
        assert order.fill_rate == 0.0

    def test_is_filled(self, filled_order, partial_order, cancelled_order):
        """Test is_filled property."""
        assert filled_order.is_filled is True
        assert partial_order.is_filled is False
        assert cancelled_order.is_filled is False

    def test_is_cancelled(self, filled_order, cancelled_order):
        """Test is_cancelled property."""
        assert filled_order.is_cancelled is False
        assert cancelled_order.is_cancelled is True

    def test_is_rejected(self):
        """Test is_rejected property."""
        rejected = OrderRecord(
            order_id="ORD",
            symbol="SPY",
            quantity=100,
            filled_quantity=0,
            side="buy",
            order_type="limit",
            status=OrderStatus.REJECTED,
            submit_time=datetime.now(timezone.utc),
            fill_time=None,
            latency_ms=0.0,
            expected_price=450.0,
            fill_price=None,
            reject_reason="Insufficient funds",
        )
        assert rejected.is_rejected is True


class TestQualityThresholds:
    """Tests for QualityThresholds dataclass."""

    def test_default_values(self):
        """Test default threshold values."""
        thresholds = QualityThresholds()

        assert thresholds.fill_rate_good == 80.0
        assert thresholds.fill_rate_warning == 50.0
        assert thresholds.slippage_good == 5.0
        assert thresholds.slippage_warning == 15.0
        assert thresholds.latency_good == 100.0
        assert thresholds.latency_warning == 500.0
        assert thresholds.cancel_rate_good == 10.0
        assert thresholds.cancel_rate_warning == 30.0

    def test_custom_values(self):
        """Test custom threshold values."""
        thresholds = QualityThresholds(
            fill_rate_good=90.0,
            slippage_good=3.0,
        )
        assert thresholds.fill_rate_good == 90.0
        assert thresholds.slippage_good == 3.0


class TestExecutionDashboard:
    """Tests for ExecutionDashboard dataclass."""

    @pytest.fixture
    def dashboard(self):
        """Create a sample dashboard."""
        now = datetime.now(timezone.utc)
        return ExecutionDashboard(
            period_start=now - timedelta(hours=1),
            period_end=now,
            total_orders=100,
            total_volume=10000,
            filled_volume=8500,
            fill_rate_pct=85.0,
            full_fill_rate_pct=75.0,
            partial_fill_rate_pct=10.0,
            avg_slippage_bps=3.5,
            median_slippage_bps=2.0,
            max_slippage_bps=15.0,
            total_slippage_cost=125.50,
            avg_latency_ms=45.0,
            median_latency_ms=40.0,
            p95_latency_ms=80.0,
            p99_latency_ms=120.0,
            cancel_rate_pct=8.0,
            reject_rate_pct=2.0,
            top_cancel_reasons=[("Timeout", 5), ("Market closed", 3)],
            top_reject_reasons=[("Insufficient funds", 2)],
            quality_score=87.5,
            fill_rate_status="good",
            slippage_status="good",
            latency_status="good",
            cancel_rate_status="good",
        )

    def test_to_dict(self, dashboard):
        """Test conversion to dictionary."""
        d = dashboard.to_dict()

        assert d["total_orders"] == 100
        assert d["fill_rate_pct"] == 85.0
        assert d["avg_slippage_bps"] == 3.5
        assert d["avg_latency_ms"] == 45.0
        assert d["quality_score"] == 87.5
        assert d["fill_rate_status"] == "good"
        assert "period_start" in d
        assert "period_end" in d


class TestExecutionQualityTracker:
    """Tests for ExecutionQualityTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a default tracker."""
        return ExecutionQualityTracker()

    @pytest.fixture
    def tracker_with_thresholds(self):
        """Create a tracker with custom thresholds."""
        return ExecutionQualityTracker(thresholds=QualityThresholds(fill_rate_good=90.0))

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.thresholds is not None
        assert tracker.max_history == 10000
        assert len(tracker.orders) == 0
        assert tracker._total_orders == 0

    def test_record_order_filled(self, tracker):
        """Test recording a filled order."""
        now = datetime.now(timezone.utc)
        record = tracker.record_order(
            order_id="ORD001",
            symbol="SPY",
            quantity=100,
            filled_quantity=100,
            side="buy",
            order_type="limit",
            status=OrderStatus.FILLED,
            submit_time=now - timedelta(milliseconds=50),
            fill_time=now,
            latency_ms=50.0,
            expected_price=450.00,
            fill_price=450.05,
        )

        assert record.order_id == "ORD001"
        assert len(tracker.orders) == 1
        assert tracker._total_orders == 1
        assert tracker._filled_orders == 1

    def test_record_order_cancelled(self, tracker):
        """Test recording a cancelled order."""
        now = datetime.now(timezone.utc)
        tracker.record_order(
            order_id="ORD001",
            symbol="SPY",
            quantity=100,
            filled_quantity=0,
            side="buy",
            order_type="limit",
            status=OrderStatus.CANCELLED,
            submit_time=now,
            fill_time=None,
            latency_ms=100.0,
            expected_price=450.00,
            cancel_reason="Timeout",
        )

        assert tracker._cancelled_orders == 1
        assert tracker._filled_orders == 0

    def test_record_order_rejected(self, tracker):
        """Test recording a rejected order."""
        now = datetime.now(timezone.utc)
        tracker.record_order(
            order_id="ORD001",
            symbol="SPY",
            quantity=100,
            filled_quantity=0,
            side="buy",
            order_type="limit",
            status=OrderStatus.REJECTED,
            submit_time=now,
            fill_time=None,
            latency_ms=0.0,
            expected_price=450.00,
            reject_reason="Insufficient funds",
        )

        assert tracker._rejected_orders == 1

    def test_get_dashboard_empty(self, tracker):
        """Test dashboard with no orders."""
        dashboard = tracker.get_dashboard()

        assert dashboard.total_orders == 0
        assert dashboard.fill_rate_pct == 0.0
        assert dashboard.quality_score == 0.0
        assert dashboard.fill_rate_status == "good"

    def test_get_dashboard_with_orders(self, tracker):
        """Test dashboard with multiple orders."""
        now = datetime.now(timezone.utc)

        # Add 8 filled orders
        for i in range(8):
            tracker.record_order(
                order_id=f"ORD{i:03d}",
                symbol="SPY",
                quantity=100,
                filled_quantity=100,
                side="buy",
                order_type="limit",
                status=OrderStatus.FILLED,
                submit_time=now - timedelta(milliseconds=50),
                fill_time=now,
                latency_ms=50.0 + i * 10,
                expected_price=450.00,
                fill_price=450.05,
            )

        # Add 2 cancelled orders
        for i in range(2):
            tracker.record_order(
                order_id=f"CAN{i:03d}",
                symbol="QQQ",
                quantity=50,
                filled_quantity=0,
                side="sell",
                order_type="limit",
                status=OrderStatus.CANCELLED,
                submit_time=now,
                fill_time=None,
                latency_ms=100.0,
                expected_price=380.00,
                cancel_reason="Timeout",
            )

        dashboard = tracker.get_dashboard()

        assert dashboard.total_orders == 10
        assert dashboard.fill_rate_pct == 80.0  # 8/10
        assert dashboard.cancel_rate_pct == 20.0  # 2/10
        assert len(dashboard.top_cancel_reasons) > 0

    def test_slippage_calculation(self, tracker):
        """Test slippage metrics calculation."""
        now = datetime.now(timezone.utc)

        # Buy order with adverse slippage (paid more)
        tracker.record_order(
            order_id="ORD001",
            symbol="SPY",
            quantity=100,
            filled_quantity=100,
            side="buy",
            order_type="limit",
            status=OrderStatus.FILLED,
            submit_time=now,
            fill_time=now,
            latency_ms=50.0,
            expected_price=100.00,
            fill_price=100.10,  # 10 bps adverse
        )

        # Sell order with adverse slippage (received less)
        tracker.record_order(
            order_id="ORD002",
            symbol="SPY",
            quantity=100,
            filled_quantity=100,
            side="sell",
            order_type="limit",
            status=OrderStatus.FILLED,
            submit_time=now,
            fill_time=now,
            latency_ms=50.0,
            expected_price=100.00,
            fill_price=99.90,  # 10 bps adverse
        )

        dashboard = tracker.get_dashboard()

        assert dashboard.avg_slippage_bps == pytest.approx(10.0, rel=0.01)
        assert dashboard.total_slippage_cost > 0

    def test_latency_metrics(self, tracker):
        """Test latency metrics calculation."""
        now = datetime.now(timezone.utc)

        latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0]

        for i, latency in enumerate(latencies):
            tracker.record_order(
                order_id=f"ORD{i:03d}",
                symbol="SPY",
                quantity=100,
                filled_quantity=100,
                side="buy",
                order_type="limit",
                status=OrderStatus.FILLED,
                submit_time=now,
                fill_time=now,
                latency_ms=latency,
                expected_price=450.00,
                fill_price=450.00,
            )

        dashboard = tracker.get_dashboard()

        # Average of latencies
        expected_avg = sum(latencies) / len(latencies)
        assert dashboard.avg_latency_ms == pytest.approx(expected_avg, rel=0.01)

        # P95 should be high
        assert dashboard.p95_latency_ms >= 400.0

    def test_quality_score_calculation(self, tracker):
        """Test quality score calculation."""
        now = datetime.now(timezone.utc)

        # Add excellent quality orders
        for i in range(10):
            tracker.record_order(
                order_id=f"ORD{i:03d}",
                symbol="SPY",
                quantity=100,
                filled_quantity=100,
                side="buy",
                order_type="limit",
                status=OrderStatus.FILLED,
                submit_time=now,
                fill_time=now,
                latency_ms=30.0,  # Good latency
                expected_price=450.00,
                fill_price=450.00,  # No slippage
            )

        dashboard = tracker.get_dashboard()

        # Should have high quality score
        assert dashboard.quality_score >= 80.0

    def test_quality_score_poor(self, tracker):
        """Test quality score with poor metrics."""
        now = datetime.now(timezone.utc)

        # Add mostly cancelled orders
        for i in range(8):
            tracker.record_order(
                order_id=f"CAN{i:03d}",
                symbol="SPY",
                quantity=100,
                filled_quantity=0,
                side="buy",
                order_type="limit",
                status=OrderStatus.CANCELLED,
                submit_time=now,
                fill_time=None,
                latency_ms=800.0,  # High latency
                expected_price=450.00,
                cancel_reason="Timeout",
            )

        # Add 2 filled with high slippage
        for i in range(2):
            tracker.record_order(
                order_id=f"ORD{i:03d}",
                symbol="SPY",
                quantity=100,
                filled_quantity=100,
                side="buy",
                order_type="limit",
                status=OrderStatus.FILLED,
                submit_time=now,
                fill_time=now,
                latency_ms=500.0,
                expected_price=450.00,
                fill_price=460.00,  # Very high slippage
            )

        dashboard = tracker.get_dashboard()

        # Should have low quality score
        assert dashboard.quality_score < 50.0

    def test_status_indicators_good(self, tracker):
        """Test status indicators for good metrics."""
        now = datetime.now(timezone.utc)

        # Add all good orders
        for i in range(10):
            tracker.record_order(
                order_id=f"ORD{i:03d}",
                symbol="SPY",
                quantity=100,
                filled_quantity=100,
                side="buy",
                order_type="limit",
                status=OrderStatus.FILLED,
                submit_time=now,
                fill_time=now,
                latency_ms=30.0,
                expected_price=450.00,
                fill_price=450.01,
            )

        dashboard = tracker.get_dashboard()

        assert dashboard.fill_rate_status == "good"
        assert dashboard.slippage_status == "good"
        assert dashboard.latency_status == "good"
        assert dashboard.cancel_rate_status == "good"

    def test_status_indicators_warning(self, tracker_with_thresholds):
        """Test status indicators at warning level."""
        now = datetime.now(timezone.utc)
        tracker = tracker_with_thresholds

        # 6 filled, 4 cancelled (60% fill rate)
        for i in range(6):
            tracker.record_order(
                order_id=f"ORD{i:03d}",
                symbol="SPY",
                quantity=100,
                filled_quantity=100,
                side="buy",
                order_type="limit",
                status=OrderStatus.FILLED,
                submit_time=now,
                fill_time=now,
                latency_ms=200.0,  # Warning latency
                expected_price=450.00,
                fill_price=450.05,
            )

        for i in range(4):
            tracker.record_order(
                order_id=f"CAN{i:03d}",
                symbol="SPY",
                quantity=100,
                filled_quantity=0,
                side="buy",
                order_type="limit",
                status=OrderStatus.CANCELLED,
                submit_time=now,
                fill_time=None,
                latency_ms=200.0,
                expected_price=450.00,
            )

        dashboard = tracker.get_dashboard()

        # With 90% threshold, 60% should be warning
        assert dashboard.fill_rate_status == "warning"

    def test_time_filtering(self, tracker):
        """Test time-based filtering."""
        now = datetime.now(timezone.utc)

        # Add old order
        tracker.record_order(
            order_id="OLD001",
            symbol="SPY",
            quantity=100,
            filled_quantity=100,
            side="buy",
            order_type="limit",
            status=OrderStatus.FILLED,
            submit_time=now - timedelta(hours=5),
            fill_time=now - timedelta(hours=5),
            latency_ms=50.0,
            expected_price=450.00,
            fill_price=450.00,
        )

        # Add recent order
        tracker.record_order(
            order_id="NEW001",
            symbol="SPY",
            quantity=100,
            filled_quantity=100,
            side="buy",
            order_type="limit",
            status=OrderStatus.FILLED,
            submit_time=now - timedelta(hours=1),
            fill_time=now - timedelta(hours=1),
            latency_ms=50.0,
            expected_price=450.00,
            fill_price=450.00,
        )

        # Filter to recent only
        start_time = now - timedelta(hours=2)
        dashboard = tracker.get_dashboard(start_time=start_time)

        assert dashboard.total_orders == 1

    def test_clear(self, tracker):
        """Test clearing order history."""
        now = datetime.now(timezone.utc)

        tracker.record_order(
            order_id="ORD001",
            symbol="SPY",
            quantity=100,
            filled_quantity=100,
            side="buy",
            order_type="limit",
            status=OrderStatus.FILLED,
            submit_time=now,
            fill_time=now,
            latency_ms=50.0,
            expected_price=450.00,
            fill_price=450.05,
        )

        assert len(tracker.orders) == 1
        assert tracker._total_orders == 1

        tracker.clear()

        assert len(tracker.orders) == 0
        assert tracker._total_orders == 0
        assert tracker._filled_orders == 0
        assert tracker._cancelled_orders == 0
        assert tracker._rejected_orders == 0

    def test_max_history_enforced(self):
        """Test that max history is enforced."""
        tracker = ExecutionQualityTracker(max_history=10)
        now = datetime.now(timezone.utc)

        for i in range(15):
            tracker.record_order(
                order_id=f"ORD{i:03d}",
                symbol="SPY",
                quantity=100,
                filled_quantity=100,
                side="buy",
                order_type="limit",
                status=OrderStatus.FILLED,
                submit_time=now,
                fill_time=now,
                latency_ms=50.0,
                expected_price=450.00,
                fill_price=450.00,
            )

        # Only keeps last 10
        assert len(tracker.orders) == 10

    def test_top_cancel_reasons(self, tracker):
        """Test tracking of cancel reasons."""
        now = datetime.now(timezone.utc)

        reasons = ["Timeout", "Timeout", "Timeout", "Market closed", "User cancelled"]

        for i, reason in enumerate(reasons):
            tracker.record_order(
                order_id=f"CAN{i:03d}",
                symbol="SPY",
                quantity=100,
                filled_quantity=0,
                side="buy",
                order_type="limit",
                status=OrderStatus.CANCELLED,
                submit_time=now,
                fill_time=None,
                latency_ms=100.0,
                expected_price=450.00,
                cancel_reason=reason,
            )

        dashboard = tracker.get_dashboard()

        # Timeout should be most common
        assert len(dashboard.top_cancel_reasons) > 0
        assert dashboard.top_cancel_reasons[0][0] == "Timeout"
        assert dashboard.top_cancel_reasons[0][1] == 3


class TestCreateExecutionTracker:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating with defaults."""
        tracker = create_execution_tracker()

        assert isinstance(tracker, ExecutionQualityTracker)
        assert tracker.thresholds.fill_rate_good == 80.0

    def test_create_with_custom_thresholds(self):
        """Test creating with custom thresholds."""
        thresholds = QualityThresholds(fill_rate_good=95.0)
        tracker = create_execution_tracker(thresholds=thresholds)

        assert tracker.thresholds.fill_rate_good == 95.0

    def test_create_with_kwargs(self):
        """Test creating with additional kwargs."""
        tracker = create_execution_tracker(max_history=500)

        assert tracker.max_history == 500


class TestGenerateExecutionReport:
    """Tests for report generation."""

    def test_report_with_orders(self):
        """Test report generation with orders."""
        tracker = create_execution_tracker()
        now = datetime.now(timezone.utc)

        for i in range(5):
            tracker.record_order(
                order_id=f"ORD{i:03d}",
                symbol="SPY",
                quantity=100,
                filled_quantity=100,
                side="buy",
                order_type="limit",
                status=OrderStatus.FILLED,
                submit_time=now,
                fill_time=now,
                latency_ms=50.0,
                expected_price=450.00,
                fill_price=450.05,
            )

        report = generate_execution_report(tracker)

        assert "EXECUTION QUALITY REPORT" in report
        assert "QUALITY SCORE" in report
        assert "FILL METRICS" in report
        assert "SLIPPAGE METRICS" in report
        assert "LATENCY METRICS" in report
        assert "Total Orders:" in report

    def test_report_empty(self):
        """Test report generation with no orders."""
        tracker = create_execution_tracker()
        report = generate_execution_report(tracker)

        assert "EXECUTION QUALITY REPORT" in report
        assert "0.0/100" in report  # Zero quality score


class TestIntegrationWithSlippageMonitor:
    """Tests for integration with SlippageMonitor."""

    def test_slippage_monitor_integration(self):
        """Test that orders are recorded to slippage monitor."""

        # Create mock slippage monitor
        class MockSlippageMonitor:
            def __init__(self):
                self.fills = []

            def record_fill(self, **kwargs):
                self.fills.append(kwargs)

        mock_monitor = MockSlippageMonitor()
        tracker = ExecutionQualityTracker(slippage_monitor=mock_monitor)
        now = datetime.now(timezone.utc)

        tracker.record_order(
            order_id="ORD001",
            symbol="SPY",
            quantity=100,
            filled_quantity=100,
            side="buy",
            order_type="limit",
            status=OrderStatus.FILLED,
            submit_time=now,
            fill_time=now,
            latency_ms=50.0,
            expected_price=450.00,
            fill_price=450.05,
        )

        # Should have been recorded to slippage monitor
        assert len(mock_monitor.fills) == 1
        assert mock_monitor.fills[0]["symbol"] == "SPY"
        assert mock_monitor.fills[0]["expected_price"] == 450.00
        assert mock_monitor.fills[0]["actual_price"] == 450.05

    def test_no_slippage_monitor_record_for_cancelled(self):
        """Test that cancelled orders don't go to slippage monitor."""

        class MockSlippageMonitor:
            def __init__(self):
                self.fills = []

            def record_fill(self, **kwargs):
                self.fills.append(kwargs)

        mock_monitor = MockSlippageMonitor()
        tracker = ExecutionQualityTracker(slippage_monitor=mock_monitor)
        now = datetime.now(timezone.utc)

        tracker.record_order(
            order_id="ORD001",
            symbol="SPY",
            quantity=100,
            filled_quantity=0,
            side="buy",
            order_type="limit",
            status=OrderStatus.CANCELLED,
            submit_time=now,
            fill_time=None,
            latency_ms=100.0,
            expected_price=450.00,
            fill_price=None,
        )

        # Should NOT have been recorded (no fill)
        assert len(mock_monitor.fills) == 0
