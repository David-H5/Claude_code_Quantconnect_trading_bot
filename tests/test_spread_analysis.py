"""
Tests for Spread Analysis Module (Sprint 5 Expansion)

Tests bid-ask spread analysis and execution cost estimation.
Part of UPGRADE-010 Sprint 5 - Quality & Test Coverage.
"""

from datetime import datetime, timedelta

import pytest

from execution.spread_analysis import (
    ExecutionCostEstimate,
    ExecutionUrgency,
    SpreadAnalyzer,
    SpreadMetrics,
    SpreadQuality,
    SpreadSnapshot,
    create_spread_analyzer,
)


class TestSpreadQuality:
    """Tests for SpreadQuality enum."""

    def test_values(self):
        """Test enum values."""
        assert SpreadQuality.EXCELLENT.value == "excellent"
        assert SpreadQuality.GOOD.value == "good"
        assert SpreadQuality.FAIR.value == "fair"
        assert SpreadQuality.POOR.value == "poor"
        assert SpreadQuality.VERY_POOR.value == "very_poor"

    def test_all_values_unique(self):
        """Test all values are unique."""
        values = [q.value for q in SpreadQuality]
        assert len(values) == len(set(values))


class TestExecutionUrgency:
    """Tests for ExecutionUrgency enum."""

    def test_values(self):
        """Test enum values."""
        assert ExecutionUrgency.PASSIVE.value == "passive"
        assert ExecutionUrgency.NORMAL.value == "normal"
        assert ExecutionUrgency.AGGRESSIVE.value == "aggressive"
        assert ExecutionUrgency.IMMEDIATE.value == "immediate"


class TestSpreadSnapshot:
    """Tests for SpreadSnapshot dataclass."""

    @pytest.fixture
    def snapshot(self):
        """Create default snapshot."""
        return SpreadSnapshot(
            timestamp=datetime.now(),
            bid_price=450.00,
            ask_price=450.10,
            bid_size=500,
            ask_size=300,
            last_price=450.05,
            volume=10000,
        )

    def test_creation(self, snapshot):
        """Test snapshot creation."""
        assert snapshot.bid_price == 450.00
        assert snapshot.ask_price == 450.10
        assert snapshot.bid_size == 500
        assert snapshot.ask_size == 300

    def test_defaults(self):
        """Test default values."""
        snapshot = SpreadSnapshot(
            timestamp=datetime.now(),
            bid_price=100.0,
            ask_price=100.1,
        )

        assert snapshot.bid_size == 0
        assert snapshot.ask_size == 0
        assert snapshot.last_price == 0.0
        assert snapshot.volume == 0

    def test_mid_price(self, snapshot):
        """Test mid-price calculation."""
        assert snapshot.mid_price == pytest.approx(450.05, rel=0.001)

    def test_spread(self, snapshot):
        """Test absolute spread."""
        assert snapshot.spread == pytest.approx(0.10, rel=0.001)

    def test_spread_pct(self, snapshot):
        """Test spread percentage."""
        # 0.10 / 450.05 = 0.000222...
        assert snapshot.spread_pct == pytest.approx(0.000222, rel=0.01)

    def test_spread_pct_zero_mid(self):
        """Test spread percentage with zero mid-price."""
        snapshot = SpreadSnapshot(
            timestamp=datetime.now(),
            bid_price=0.0,
            ask_price=0.0,
        )
        assert snapshot.spread_pct == 0.0

    def test_spread_bps(self, snapshot):
        """Test spread in basis points."""
        # 0.000222 * 10000 = 2.22 bps
        assert snapshot.spread_bps == pytest.approx(2.22, rel=0.01)

    def test_imbalance_positive(self, snapshot):
        """Test positive imbalance (more bid pressure)."""
        # (500 - 300) / 800 = 0.25
        assert snapshot.imbalance == pytest.approx(0.25, rel=0.01)

    def test_imbalance_negative(self):
        """Test negative imbalance (more ask pressure)."""
        snapshot = SpreadSnapshot(
            timestamp=datetime.now(),
            bid_price=100.0,
            ask_price=100.1,
            bid_size=200,
            ask_size=600,
        )
        # (200 - 600) / 800 = -0.5
        assert snapshot.imbalance == pytest.approx(-0.5, rel=0.01)

    def test_imbalance_zero_size(self):
        """Test imbalance with zero sizes."""
        snapshot = SpreadSnapshot(
            timestamp=datetime.now(),
            bid_price=100.0,
            ask_price=100.1,
        )
        assert snapshot.imbalance == 0.0

    def test_get_quality_excellent(self):
        """Test excellent quality classification (<0.05%)."""
        snapshot = SpreadSnapshot(
            timestamp=datetime.now(),
            bid_price=100.00,
            ask_price=100.04,  # 0.04% spread
        )
        assert snapshot.get_quality() == SpreadQuality.EXCELLENT

    def test_get_quality_good(self):
        """Test good quality classification (0.05-0.1%)."""
        snapshot = SpreadSnapshot(
            timestamp=datetime.now(),
            bid_price=100.00,
            ask_price=100.07,  # 0.07% spread
        )
        assert snapshot.get_quality() == SpreadQuality.GOOD

    def test_get_quality_fair(self):
        """Test fair quality classification (0.1-0.3%)."""
        snapshot = SpreadSnapshot(
            timestamp=datetime.now(),
            bid_price=100.00,
            ask_price=100.20,  # 0.2% spread
        )
        assert snapshot.get_quality() == SpreadQuality.FAIR

    def test_get_quality_poor(self):
        """Test poor quality classification (0.3-0.5%)."""
        snapshot = SpreadSnapshot(
            timestamp=datetime.now(),
            bid_price=100.00,
            ask_price=100.40,  # 0.4% spread
        )
        assert snapshot.get_quality() == SpreadQuality.POOR

    def test_get_quality_very_poor(self):
        """Test very poor quality classification (>0.5%)."""
        snapshot = SpreadSnapshot(
            timestamp=datetime.now(),
            bid_price=100.00,
            ask_price=101.00,  # 1% spread
        )
        assert snapshot.get_quality() == SpreadQuality.VERY_POOR


class TestExecutionCostEstimate:
    """Tests for ExecutionCostEstimate dataclass."""

    def test_creation(self):
        """Test estimate creation."""
        estimate = ExecutionCostEstimate(
            half_spread_cost=5.0,
            market_impact=2.0,
            timing_cost=1.0,
            total_cost=8.0,
            cost_bps=3.5,
            recommended_order_type="limit",
            confidence=0.85,
        )

        assert estimate.half_spread_cost == 5.0
        assert estimate.market_impact == 2.0
        assert estimate.total_cost == 8.0
        assert estimate.recommended_order_type == "limit"
        assert estimate.confidence == 0.85

    def test_default_confidence(self):
        """Test default confidence value."""
        estimate = ExecutionCostEstimate(
            half_spread_cost=0,
            market_impact=0,
            timing_cost=0,
            total_cost=0,
            cost_bps=0,
            recommended_order_type="limit",
        )
        assert estimate.confidence == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        estimate = ExecutionCostEstimate(
            half_spread_cost=5.0,
            market_impact=2.0,
            timing_cost=1.0,
            total_cost=8.0,
            cost_bps=3.5,
            recommended_order_type="limit",
            confidence=0.85,
        )

        d = estimate.to_dict()

        assert d["half_spread_cost"] == 5.0
        assert d["market_impact"] == 2.0
        assert d["timing_cost"] == 1.0
        assert d["total_cost"] == 8.0
        assert d["cost_bps"] == 3.5
        assert d["recommended_order_type"] == "limit"
        assert d["confidence"] == 0.85


class TestSpreadMetrics:
    """Tests for SpreadMetrics dataclass."""

    def test_defaults(self):
        """Test default values."""
        metrics = SpreadMetrics()

        assert metrics.avg_spread_bps == 0.0
        assert metrics.min_spread_bps == 0.0
        assert metrics.max_spread_bps == 0.0
        assert metrics.std_spread_bps == 0.0
        assert metrics.avg_imbalance == 0.0
        assert metrics.samples == 0
        assert metrics.quality_distribution == {}

    def test_creation_with_values(self):
        """Test creation with values."""
        metrics = SpreadMetrics(
            avg_spread_bps=5.5,
            min_spread_bps=2.0,
            max_spread_bps=10.0,
            std_spread_bps=2.5,
            avg_imbalance=0.15,
            avg_bid_size=500.0,
            avg_ask_size=450.0,
            samples=100,
            quality_distribution={"good": 50, "fair": 50},
        )

        assert metrics.avg_spread_bps == 5.5
        assert metrics.min_spread_bps == 2.0
        assert metrics.max_spread_bps == 10.0
        assert metrics.samples == 100

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = SpreadMetrics(
            avg_spread_bps=5.0,
            samples=50,
            quality_distribution={"good": 30, "fair": 20},
        )

        d = metrics.to_dict()

        assert d["avg_spread_bps"] == 5.0
        assert d["samples"] == 50
        assert d["quality_distribution"]["good"] == 30


class TestSpreadAnalyzer:
    """Tests for SpreadAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create default analyzer."""
        return SpreadAnalyzer(max_history=100)

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.max_history == 100
        assert analyzer.impact_coefficient == 0.1
        assert len(analyzer.history) == 0

    def test_custom_parameters(self):
        """Test with custom parameters."""
        analyzer = SpreadAnalyzer(max_history=500, impact_coefficient=0.2)
        assert analyzer.max_history == 500
        assert analyzer.impact_coefficient == 0.2

    def test_update_creates_snapshot(self, analyzer):
        """Test update creates snapshot."""
        now = datetime.now()
        snapshot = analyzer.update(
            symbol="SPY",
            bid=450.00,
            ask=450.10,
            bid_size=500,
            ask_size=300,
            timestamp=now,
        )

        assert snapshot.bid_price == 450.00
        assert snapshot.ask_price == 450.10
        assert "SPY" in analyzer.history
        assert len(analyzer.history["SPY"]) == 1

    def test_update_default_timestamp(self, analyzer):
        """Test update with default timestamp."""
        snapshot = analyzer.update(
            symbol="SPY",
            bid=450.00,
            ask=450.10,
        )

        assert snapshot.timestamp is not None

    def test_update_trims_history(self):
        """Test history is trimmed to max size."""
        analyzer = SpreadAnalyzer(max_history=5)

        for i in range(10):
            analyzer.update(
                symbol="SPY",
                bid=450.00 + i * 0.01,
                ask=450.10 + i * 0.01,
            )

        assert len(analyzer.history["SPY"]) == 5

    def test_get_current_spread(self, analyzer):
        """Test getting current spread."""
        analyzer.update(symbol="SPY", bid=450.00, ask=450.10)
        analyzer.update(symbol="SPY", bid=450.05, ask=450.15)

        current = analyzer.get_current_spread("SPY")

        assert current is not None
        assert current.bid_price == 450.05
        assert current.ask_price == 450.15

    def test_get_current_spread_no_data(self, analyzer):
        """Test getting current spread with no data."""
        current = analyzer.get_current_spread("UNKNOWN")
        assert current is None

    def test_get_spread_metrics(self, analyzer):
        """Test spread metrics calculation."""
        now = datetime.now()

        # Add multiple snapshots
        for i in range(5):
            analyzer.update(
                symbol="SPY",
                bid=450.00,
                ask=450.10 + i * 0.02,  # Varying spread
                bid_size=500,
                ask_size=300,
                timestamp=now - timedelta(minutes=i),
            )

        metrics = analyzer.get_spread_metrics("SPY", lookback_minutes=60)

        assert metrics.samples == 5
        assert metrics.avg_spread_bps > 0
        assert metrics.min_spread_bps <= metrics.max_spread_bps

    def test_get_spread_metrics_no_data(self, analyzer):
        """Test spread metrics with no data."""
        metrics = analyzer.get_spread_metrics("UNKNOWN")

        assert metrics.samples == 0
        assert metrics.avg_spread_bps == 0.0

    def test_get_spread_metrics_no_recent_data(self, analyzer):
        """Test spread metrics with old data only."""
        old_time = datetime.now() - timedelta(hours=2)

        analyzer.update(
            symbol="SPY",
            bid=450.00,
            ask=450.10,
            timestamp=old_time,
        )

        metrics = analyzer.get_spread_metrics("SPY", lookback_minutes=60)

        assert metrics.samples == 0

    def test_estimate_execution_cost_no_data(self, analyzer):
        """Test cost estimate with no data."""
        estimate = analyzer.estimate_execution_cost(
            symbol="UNKNOWN",
            quantity=100,
            side="buy",
        )

        assert estimate.total_cost == 0
        assert estimate.confidence == 0

    def test_estimate_execution_cost_buy(self, analyzer):
        """Test cost estimate for buy order."""
        analyzer.update(
            symbol="SPY",
            bid=450.00,
            ask=450.10,
            bid_size=500,
            ask_size=300,
        )

        estimate = analyzer.estimate_execution_cost(
            symbol="SPY",
            quantity=100,
            side="buy",
        )

        assert estimate.half_spread_cost > 0
        assert estimate.total_cost > 0
        assert estimate.cost_bps > 0

    def test_estimate_execution_cost_sell(self, analyzer):
        """Test cost estimate for sell order."""
        analyzer.update(
            symbol="SPY",
            bid=450.00,
            ask=450.10,
            bid_size=500,
            ask_size=300,
        )

        estimate = analyzer.estimate_execution_cost(
            symbol="SPY",
            quantity=100,
            side="sell",
        )

        assert estimate.total_cost > 0

    def test_estimate_execution_cost_with_adv(self, analyzer):
        """Test cost estimate with average daily volume."""
        analyzer.update(
            symbol="SPY",
            bid=450.00,
            ask=450.10,
            bid_size=500,
            ask_size=300,
        )

        estimate = analyzer.estimate_execution_cost(
            symbol="SPY",
            quantity=10000,
            side="buy",
            average_daily_volume=1000000,  # ADV = 1M shares
        )

        # Should have market impact with ADV
        assert estimate.market_impact > 0

    def test_estimate_execution_cost_exceeds_liquidity(self, analyzer):
        """Test cost estimate when order exceeds visible liquidity."""
        analyzer.update(
            symbol="SPY",
            bid=450.00,
            ask=450.10,
            bid_size=100,
            ask_size=100,
        )

        estimate = analyzer.estimate_execution_cost(
            symbol="SPY",
            quantity=500,  # 5x visible size
            side="buy",
        )

        assert estimate.market_impact > 0

    def test_estimate_execution_cost_passive_urgency(self, analyzer):
        """Test cost estimate with passive urgency."""
        analyzer.update(symbol="SPY", bid=450.00, ask=450.10)

        estimate = analyzer.estimate_execution_cost(
            symbol="SPY",
            quantity=100,
            side="buy",
            urgency=ExecutionUrgency.PASSIVE,
        )

        assert estimate.timing_cost == 0
        assert estimate.recommended_order_type == "limit"

    def test_estimate_execution_cost_immediate_urgency(self, analyzer):
        """Test cost estimate with immediate urgency."""
        analyzer.update(symbol="SPY", bid=450.00, ask=450.10)

        estimate = analyzer.estimate_execution_cost(
            symbol="SPY",
            quantity=100,
            side="buy",
            urgency=ExecutionUrgency.IMMEDIATE,
        )

        assert estimate.timing_cost > 0
        assert estimate.recommended_order_type == "market"

    def test_estimate_execution_cost_aggressive_urgency(self, analyzer):
        """Test cost estimate with aggressive urgency on wide spread."""
        # Wide spread = poor quality
        analyzer.update(symbol="SPY", bid=450.00, ask=452.00)

        estimate = analyzer.estimate_execution_cost(
            symbol="SPY",
            quantity=100,
            side="buy",
            urgency=ExecutionUrgency.AGGRESSIVE,
        )

        assert estimate.recommended_order_type == "market"

    def test_get_optimal_limit_price_buy(self, analyzer):
        """Test optimal limit price for buy orders."""
        analyzer.update(symbol="SPY", bid=450.00, ask=450.10)

        # Passive - post at bid
        price = analyzer.get_optimal_limit_price("SPY", "buy", ExecutionUrgency.PASSIVE)
        assert price == 450.00

        # Immediate - take the ask
        price = analyzer.get_optimal_limit_price("SPY", "buy", ExecutionUrgency.IMMEDIATE)
        assert price == 450.10

        # Aggressive - close to ask
        price = analyzer.get_optimal_limit_price("SPY", "buy", ExecutionUrgency.AGGRESSIVE)
        assert price > 450.05 and price < 450.10

        # Normal - slightly above mid
        price = analyzer.get_optimal_limit_price("SPY", "buy", ExecutionUrgency.NORMAL)
        assert price > 450.05 and price < 450.08

    def test_get_optimal_limit_price_sell(self, analyzer):
        """Test optimal limit price for sell orders."""
        analyzer.update(symbol="SPY", bid=450.00, ask=450.10)

        # Passive - post at ask
        price = analyzer.get_optimal_limit_price("SPY", "sell", ExecutionUrgency.PASSIVE)
        assert price == 450.10

        # Immediate - take the bid
        price = analyzer.get_optimal_limit_price("SPY", "sell", ExecutionUrgency.IMMEDIATE)
        assert price == 450.00

        # Aggressive - close to bid
        price = analyzer.get_optimal_limit_price("SPY", "sell", ExecutionUrgency.AGGRESSIVE)
        assert price > 450.00 and price < 450.05

        # Normal - slightly below mid
        price = analyzer.get_optimal_limit_price("SPY", "sell", ExecutionUrgency.NORMAL)
        assert price > 450.02 and price < 450.05

    def test_get_optimal_limit_price_no_data(self, analyzer):
        """Test optimal limit price with no data."""
        price = analyzer.get_optimal_limit_price("UNKNOWN", "buy")
        assert price is None

    def test_is_execution_favorable_no_data(self, analyzer):
        """Test favorable check with no data."""
        favorable, reason = analyzer.is_execution_favorable("UNKNOWN", "buy")

        assert not favorable
        assert "No spread data" in reason

    def test_is_execution_favorable_tight_spread(self, analyzer):
        """Test favorable with tight spread."""
        now = datetime.now()
        for i in range(15):
            analyzer.update(
                symbol="SPY",
                bid=450.00,
                ask=450.05,  # 1.1 bps spread
                bid_size=500,
                ask_size=500,
                timestamp=now - timedelta(minutes=i),
            )

        favorable, reason = analyzer.is_execution_favorable("SPY", "buy", threshold_bps=5.0)

        assert favorable
        assert "Favorable" in reason

    def test_is_execution_favorable_wide_spread(self, analyzer):
        """Test unfavorable with wide spread."""
        analyzer.update(
            symbol="SPY",
            bid=450.00,
            ask=451.00,  # ~22 bps spread
        )

        favorable, reason = analyzer.is_execution_favorable("SPY", "buy", threshold_bps=10.0)

        assert not favorable
        assert "Spread too wide" in reason

    def test_is_execution_favorable_wider_than_average(self, analyzer):
        """Test unfavorable when wider than average."""
        now = datetime.now()

        # Add history with tight spreads
        for i in range(15):
            analyzer.update(
                symbol="SPY",
                bid=450.00,
                ask=450.03,  # ~0.67 bps
                timestamp=now - timedelta(minutes=i + 1),
            )

        # Current spread much wider
        analyzer.update(
            symbol="SPY",
            bid=450.00,
            ask=450.08,  # ~1.8 bps (>1.5x average)
            timestamp=now,
        )

        favorable, reason = analyzer.is_execution_favorable("SPY", "buy", threshold_bps=20.0)

        assert not favorable
        assert "wider than average" in reason

    def test_is_execution_favorable_sell_pressure(self, analyzer):
        """Test unfavorable with heavy sell pressure for buy."""
        now = datetime.now()
        for i in range(15):
            analyzer.update(
                symbol="SPY",
                bid=450.00,
                ask=450.05,
                bid_size=100,
                ask_size=500,  # Heavy ask side
                timestamp=now - timedelta(minutes=i),
            )

        favorable, reason = analyzer.is_execution_favorable("SPY", "buy")

        assert not favorable
        assert "sell pressure" in reason

    def test_is_execution_favorable_buy_pressure(self, analyzer):
        """Test unfavorable with heavy buy pressure for sell."""
        now = datetime.now()
        for i in range(15):
            analyzer.update(
                symbol="SPY",
                bid=450.00,
                ask=450.05,
                bid_size=500,
                ask_size=100,  # Heavy bid side
                timestamp=now - timedelta(minutes=i),
            )

        favorable, reason = analyzer.is_execution_favorable("SPY", "sell")

        assert not favorable
        assert "buy pressure" in reason

    def test_get_liquidity_score_no_data(self, analyzer):
        """Test liquidity score with no data."""
        score = analyzer.get_liquidity_score("UNKNOWN")
        assert score == 0.0

    def test_get_liquidity_score_tight_spread(self, analyzer):
        """Test liquidity score with tight spread."""
        analyzer.update(
            symbol="SPY",
            bid=450.00,
            ask=450.01,  # Very tight spread
            bid_size=5000,
            ask_size=5000,  # Good depth
        )

        score = analyzer.get_liquidity_score("SPY")

        # Should be high (tight spread + good depth)
        assert score > 80

    def test_get_liquidity_score_wide_spread(self, analyzer):
        """Test liquidity score with wide spread."""
        analyzer.update(
            symbol="SPY",
            bid=450.00,
            ask=451.00,  # Wide spread (~22 bps)
            bid_size=100,
            ask_size=100,  # Low depth
        )

        score = analyzer.get_liquidity_score("SPY")

        # Should be low
        assert score < 30

    def test_get_summary(self, analyzer):
        """Test summary generation."""
        analyzer.update(
            symbol="SPY",
            bid=450.00,
            ask=450.10,
            bid_size=500,
            ask_size=300,
        )

        summary = analyzer.get_summary("SPY")

        assert summary["symbol"] == "SPY"
        assert "current" in summary
        assert summary["current"]["bid"] == 450.00
        assert summary["current"]["ask"] == 450.10
        assert "metrics" in summary
        assert "liquidity_score" in summary

    def test_get_summary_no_data(self, analyzer):
        """Test summary with no data."""
        summary = analyzer.get_summary("UNKNOWN")

        assert summary["symbol"] == "UNKNOWN"
        assert "error" in summary


class TestCreateSpreadAnalyzer:
    """Tests for factory function."""

    def test_default_parameters(self):
        """Test factory with defaults."""
        analyzer = create_spread_analyzer()

        assert analyzer.max_history == 1000
        assert analyzer.impact_coefficient == 0.1

    def test_custom_parameters(self):
        """Test factory with custom parameters."""
        analyzer = create_spread_analyzer(
            max_history=500,
            impact_coefficient=0.15,
        )

        assert analyzer.max_history == 500
        assert analyzer.impact_coefficient == 0.15


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_options_spread_analysis(self):
        """Test spread analysis for options trading."""
        analyzer = SpreadAnalyzer()
        now = datetime.now()

        # Simulate option chain quotes
        symbols = ["SPY_240119C450", "SPY_240119C455", "SPY_240119P445"]
        spreads = [
            (5.00, 5.20),  # 20 cent spread on ITM call
            (2.50, 2.80),  # 30 cent spread on OTM call
            (1.80, 2.10),  # 30 cent spread on OTM put
        ]

        for symbol, (bid, ask) in zip(symbols, spreads):
            analyzer.update(
                symbol=symbol,
                bid=bid,
                ask=ask,
                bid_size=50,
                ask_size=50,
                timestamp=now,
            )

        # Get summaries
        for symbol in symbols:
            summary = analyzer.get_summary(symbol)
            assert summary["current"]["spread_bps"] > 0

    def test_high_frequency_updates(self):
        """Test handling high frequency quote updates."""
        analyzer = SpreadAnalyzer(max_history=100)
        now = datetime.now()

        # Simulate 200 quote updates
        for i in range(200):
            bid = 450.00 + (i % 10) * 0.01
            ask = bid + 0.10
            analyzer.update(
                symbol="SPY",
                bid=bid,
                ask=ask,
                timestamp=now + timedelta(milliseconds=i * 100),
            )

        # Should only keep last 100
        assert len(analyzer.history["SPY"]) == 100

        # Metrics should work
        metrics = analyzer.get_spread_metrics("SPY")
        assert metrics.samples <= 100

    def test_execution_decision_workflow(self):
        """Test complete execution decision workflow."""
        analyzer = SpreadAnalyzer()
        now = datetime.now()

        # Populate with recent data
        for i in range(30):
            analyzer.update(
                symbol="AAPL",
                bid=180.00,
                ask=180.05,
                bid_size=300 + i * 10,
                ask_size=250 + i * 5,
                timestamp=now - timedelta(minutes=30 - i),
            )

        # Check if favorable
        favorable, reason = analyzer.is_execution_favorable("AAPL", "buy", threshold_bps=5.0)

        if favorable:
            # Get optimal price
            price = analyzer.get_optimal_limit_price("AAPL", "buy", ExecutionUrgency.NORMAL)
            assert price is not None

            # Estimate costs
            estimate = analyzer.estimate_execution_cost(
                symbol="AAPL",
                quantity=100,
                side="buy",
                urgency=ExecutionUrgency.NORMAL,
            )
            assert estimate.confidence > 0.5
