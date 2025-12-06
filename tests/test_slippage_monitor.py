"""
Tests for Slippage Monitor

Tests verify the slippage monitor correctly:
- Calculates slippage in basis points
- Tracks fills and generates statistics
- Generates alerts on high slippage
- Provides per-symbol analysis
"""

from datetime import datetime, timedelta, timezone

import pytest

from execution.slippage_monitor import (
    AlertLevel,
    SlippageAlert,
    SlippageDirection,
    SlippageMonitor,
    create_slippage_monitor,
    generate_slippage_report,
)


@pytest.fixture
def monitor():
    """Create a slippage monitor with default settings."""
    return SlippageMonitor(
        alert_threshold_bps=10.0,
        warning_threshold_bps=5.0,
        critical_threshold_bps=25.0,
    )


class TestSlippageCalculation:
    """Tests for slippage calculation."""

    def test_buy_adverse_slippage(self, monitor):
        """Buy order filled above expected price is adverse."""
        record = monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=450.45,  # Paid more
            quantity=100,
            side="buy",
        )

        assert record.direction == SlippageDirection.ADVERSE
        assert record.slippage_bps == pytest.approx(10.0, abs=0.5)  # ~10 bps

    def test_buy_favorable_slippage(self, monitor):
        """Buy order filled below expected price is favorable."""
        record = monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=449.55,  # Paid less
            quantity=100,
            side="buy",
        )

        assert record.direction == SlippageDirection.FAVORABLE
        assert record.slippage_bps == pytest.approx(10.0, abs=0.5)

    def test_sell_adverse_slippage(self, monitor):
        """Sell order filled below expected price is adverse."""
        record = monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=449.55,  # Received less
            quantity=100,
            side="sell",
        )

        assert record.direction == SlippageDirection.ADVERSE
        assert record.slippage_bps == pytest.approx(10.0, abs=0.5)

    def test_sell_favorable_slippage(self, monitor):
        """Sell order filled above expected price is favorable."""
        record = monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=450.45,  # Received more
            quantity=100,
            side="sell",
        )

        assert record.direction == SlippageDirection.FAVORABLE
        assert record.slippage_bps == pytest.approx(10.0, abs=0.5)

    def test_neutral_slippage(self, monitor):
        """Fill at expected price is neutral."""
        record = monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=450.00,
            quantity=100,
            side="buy",
        )

        assert record.direction == SlippageDirection.NEUTRAL
        assert record.slippage_bps == pytest.approx(0.0, abs=0.1)

    def test_slippage_bps_calculation(self, monitor):
        """Verify basis points calculation."""
        # 1 basis point = 0.01% = 0.0001
        # $450 * 0.0001 = $0.045
        # So $0.45 diff on $450 = 10 bps
        record = monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=450.45,
            quantity=100,
            side="buy",
        )

        # 0.45 / 450 = 0.001 = 10 bps
        assert record.slippage_bps == pytest.approx(10.0, abs=0.1)


class TestAlertGeneration:
    """Tests for alert generation."""

    def test_critical_alert(self, monitor):
        """Critical alert generated for very high slippage."""
        # 30 bps slippage (above 25 critical threshold)
        monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=451.35,  # ~30 bps
            quantity=100,
            side="buy",
        )

        alerts = monitor.get_recent_alerts()
        assert len(alerts) >= 1

        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical_alerts) >= 1

    def test_warning_alert(self, monitor):
        """Warning alert generated for high slippage."""
        # 15 bps slippage (above 10 alert threshold, below 25 critical)
        monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=450.675,  # ~15 bps
            quantity=100,
            side="buy",
        )

        alerts = monitor.get_recent_alerts()
        warning_alerts = [a for a in alerts if a.level == AlertLevel.WARNING]
        assert len(warning_alerts) >= 1

    def test_info_alert(self, monitor):
        """Info alert generated for moderate slippage."""
        # 7 bps slippage (above 5 warning threshold, below 10 alert)
        monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=450.315,  # ~7 bps
            quantity=100,
            side="buy",
        )

        alerts = monitor.get_recent_alerts()
        info_alerts = [a for a in alerts if a.level == AlertLevel.INFO]
        assert len(info_alerts) >= 1

    def test_no_alert_for_favorable_slippage(self, monitor):
        """No alert for favorable slippage."""
        monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=449.00,  # ~22 bps favorable
            quantity=100,
            side="buy",
        )

        # Should not generate alerts for favorable slippage
        assert len(monitor.alerts) == 0

    def test_alert_callback(self):
        """Alert callback is invoked."""
        received_alerts: list[SlippageAlert] = []

        def callback(alert: SlippageAlert):
            received_alerts.append(alert)

        monitor = SlippageMonitor(
            alert_threshold_bps=10.0,
            alert_callback=callback,
        )

        monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=451.35,  # ~30 bps adverse
            quantity=100,
            side="buy",
        )

        assert len(received_alerts) >= 1


class TestMetricsAggregation:
    """Tests for metrics aggregation."""

    def test_empty_metrics(self, monitor):
        """Metrics for empty monitor."""
        metrics = monitor.get_metrics()

        assert metrics.total_fills == 0
        assert metrics.avg_slippage_bps == 0.0

    def test_aggregate_statistics(self, monitor):
        """Aggregate statistics calculated correctly."""
        # Record multiple fills
        fills = [
            ("ORD001", 450.00, 450.10, 100),  # ~2.2 bps
            ("ORD002", 450.00, 450.20, 100),  # ~4.4 bps
            ("ORD003", 450.00, 450.30, 100),  # ~6.7 bps
            ("ORD004", 450.00, 450.40, 100),  # ~8.9 bps
            ("ORD005", 450.00, 450.50, 100),  # ~11.1 bps
        ]

        for order_id, expected, actual, qty in fills:
            monitor.record_fill(
                order_id=order_id,
                symbol="SPY",
                expected_price=expected,
                actual_price=actual,
                quantity=qty,
                side="buy",
            )

        metrics = monitor.get_metrics()

        assert metrics.total_fills == 5
        assert metrics.total_volume == 500
        assert metrics.avg_slippage_bps > 0
        assert metrics.max_slippage_bps > metrics.min_slippage_bps

    def test_symbol_breakdown(self, monitor):
        """Metrics include symbol breakdown."""
        # Record fills for multiple symbols
        monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=450.20,
            quantity=100,
            side="buy",
        )

        monitor.record_fill(
            order_id="ORD002",
            symbol="AAPL",
            expected_price=175.00,
            actual_price=175.10,
            quantity=50,
            side="buy",
        )

        metrics = monitor.get_metrics()

        assert "SPY" in metrics.by_symbol
        assert "AAPL" in metrics.by_symbol
        assert metrics.by_symbol["SPY"].fill_count == 1
        assert metrics.by_symbol["AAPL"].fill_count == 1

    def test_time_filtered_metrics(self, monitor):
        """Metrics can be filtered by time."""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)

        # Record old fill
        monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=450.20,
            quantity=100,
            side="buy",
            timestamp=hour_ago - timedelta(hours=1),
        )

        # Record recent fill
        monitor.record_fill(
            order_id="ORD002",
            symbol="SPY",
            expected_price=450.00,
            actual_price=450.30,
            quantity=100,
            side="buy",
            timestamp=now,
        )

        # Get metrics for last hour only
        metrics = monitor.get_metrics(start_time=hour_ago)

        assert metrics.total_fills == 1

    def test_direction_percentages(self, monitor):
        """Direction percentages calculated correctly."""
        # 2 favorable, 1 adverse, 1 neutral
        monitor.record_fill("ORD001", "SPY", 450.00, 449.50, 100, "buy")  # favorable
        monitor.record_fill("ORD002", "SPY", 450.00, 449.55, 100, "buy")  # favorable
        monitor.record_fill("ORD003", "SPY", 450.00, 450.50, 100, "buy")  # adverse
        monitor.record_fill("ORD004", "SPY", 450.00, 450.00, 100, "buy")  # neutral

        metrics = monitor.get_metrics()

        assert metrics.favorable_pct == pytest.approx(50.0, abs=1.0)
        assert metrics.adverse_pct == pytest.approx(25.0, abs=1.0)
        assert metrics.neutral_pct == pytest.approx(25.0, abs=1.0)


class TestSymbolStats:
    """Tests for per-symbol statistics."""

    def test_symbol_stats(self, monitor):
        """Can get stats for specific symbol."""
        for i in range(5):
            monitor.record_fill(
                order_id=f"ORD{i}",
                symbol="SPY",
                expected_price=450.00,
                actual_price=450.00 + (i * 0.05),
                quantity=100,
                side="buy",
            )

        stats = monitor.get_symbol_stats("SPY")

        assert stats.fill_count == 5
        assert stats.total_volume == 500
        assert stats.avg_slippage_bps > 0

    def test_empty_symbol_stats(self, monitor):
        """Empty stats for symbol with no fills."""
        stats = monitor.get_symbol_stats("UNKNOWN")

        assert stats.fill_count == 0
        assert stats.avg_slippage_bps == 0.0


class TestWorstFills:
    """Tests for worst fills retrieval."""

    def test_get_worst_fills(self, monitor):
        """Can get fills with worst slippage."""
        # Record fills with varying slippage
        slippages = [
            ("ORD001", 450.50),  # ~11 bps
            ("ORD002", 450.10),  # ~2 bps
            ("ORD003", 451.00),  # ~22 bps
            ("ORD004", 450.30),  # ~7 bps
            ("ORD005", 450.80),  # ~18 bps
        ]

        for order_id, actual in slippages:
            monitor.record_fill(
                order_id=order_id,
                symbol="SPY",
                expected_price=450.00,
                actual_price=actual,
                quantity=100,
                side="buy",
            )

        worst = monitor.get_worst_fills(count=3)

        # Should return top 3 worst
        assert len(worst) == 3
        # Should be sorted by slippage (worst first)
        assert worst[0].slippage_bps >= worst[1].slippage_bps
        assert worst[1].slippage_bps >= worst[2].slippage_bps


class TestFillRecord:
    """Tests for FillRecord."""

    def test_to_dict(self, monitor):
        """FillRecord can be serialized."""
        record = monitor.record_fill(
            order_id="ORD001",
            symbol="SPY",
            expected_price=450.00,
            actual_price=450.20,
            quantity=100,
            side="buy",
            order_type="limit",
            metadata={"strategy": "momentum"},
        )

        record_dict = record.to_dict()

        assert record_dict["order_id"] == "ORD001"
        assert record_dict["symbol"] == "SPY"
        assert record_dict["order_type"] == "limit"
        assert "timestamp" in record_dict
        assert record_dict["metadata"]["strategy"] == "momentum"


class TestClearAndReset:
    """Tests for clear and reset operations."""

    def test_clear_history(self, monitor):
        """Can clear fill history."""
        monitor.record_fill("ORD001", "SPY", 450.00, 450.20, 100, "buy")
        monitor.record_fill("ORD002", "SPY", 450.00, 450.30, 100, "buy")

        assert len(monitor.fill_history) == 2

        monitor.clear_history()

        assert len(monitor.fill_history) == 0
        assert len(monitor.alerts) == 0

    def test_reset_alerts_only(self, monitor):
        """Can reset alerts while keeping history."""
        monitor.record_fill("ORD001", "SPY", 450.00, 451.50, 100, "buy")  # High slippage

        assert len(monitor.fill_history) == 1
        assert len(monitor.alerts) >= 1

        monitor.reset_alerts()

        assert len(monitor.fill_history) == 1  # History kept
        assert len(monitor.alerts) == 0  # Alerts cleared


class TestFactoryAndReport:
    """Tests for factory function and report generation."""

    def test_create_slippage_monitor(self):
        """Factory creates monitor."""
        monitor = create_slippage_monitor(
            alert_threshold_bps=15.0,
            warning_threshold_bps=8.0,
        )

        assert monitor.alert_threshold == 15.0
        assert monitor.warning_threshold == 8.0

    def test_generate_report(self, monitor):
        """Generates readable report."""
        # Record some fills
        for i in range(5):
            monitor.record_fill(
                order_id=f"ORD{i}",
                symbol="SPY",
                expected_price=450.00,
                actual_price=450.00 + (i * 0.1),
                quantity=100,
                side="buy",
            )

        report = generate_slippage_report(monitor)

        assert "SLIPPAGE ANALYSIS REPORT" in report
        assert "SUMMARY" in report
        assert "Total Fills: 5" in report
        assert "Average:" in report
        assert "BY SYMBOL" in report


class TestMaxHistory:
    """Tests for history size limits."""

    def test_max_history_enforced(self):
        """History doesn't exceed max size."""
        monitor = SlippageMonitor(max_history=5)

        for i in range(10):
            monitor.record_fill(
                order_id=f"ORD{i}",
                symbol="SPY",
                expected_price=450.00,
                actual_price=450.20,
                quantity=100,
                side="buy",
            )

        assert len(monitor.fill_history) == 5
