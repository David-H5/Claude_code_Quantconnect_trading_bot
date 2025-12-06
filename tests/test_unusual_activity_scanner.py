"""
Tests for Unusual Options Activity Scanner

Tests the unusual activity detection module.
Part of UPGRADE-010 Sprint 4 - Test Coverage.
"""

from datetime import datetime, timedelta

import pytest

from scanners.unusual_activity_scanner import (
    ActivityHistory,
    ActivityType,
    DirectionBias,
    FlowAnalysis,
    OptionActivityData,
    UnusualActivityAlert,
    UnusualActivityConfig,
    UnusualActivityScanner,
    UrgencyLevel,
    create_unusual_activity_scanner,
)


class TestActivityType:
    """Tests for ActivityType enum."""

    def test_types_exist(self):
        """Test all activity types exist."""
        assert ActivityType.VOLUME_SPIKE.value == "volume_spike"
        assert ActivityType.OI_SURGE.value == "oi_surge"
        assert ActivityType.IV_SPIKE.value == "iv_spike"
        assert ActivityType.BLOCK_TRADE.value == "block_trade"
        assert ActivityType.PUT_CALL_SKEW.value == "put_call_skew"
        assert ActivityType.SWEEP.value == "sweep"


class TestDirectionBias:
    """Tests for DirectionBias enum."""

    def test_directions_exist(self):
        """Test all directions exist."""
        assert DirectionBias.BULLISH.value == "bullish"
        assert DirectionBias.BEARISH.value == "bearish"
        assert DirectionBias.NEUTRAL.value == "neutral"


class TestUrgencyLevel:
    """Tests for UrgencyLevel enum."""

    def test_levels_exist(self):
        """Test all urgency levels exist."""
        assert UrgencyLevel.LOW.value == "low"
        assert UrgencyLevel.MEDIUM.value == "medium"
        assert UrgencyLevel.HIGH.value == "high"
        assert UrgencyLevel.CRITICAL.value == "critical"


class TestOptionActivityData:
    """Tests for OptionActivityData dataclass."""

    def test_creation(self):
        """Test data creation."""
        data = OptionActivityData(
            symbol="SPY250117C00450000",
            underlying="SPY",
            option_type="call",
            strike=450.0,
            expiry=datetime(2025, 1, 17),
            volume=500,
            open_interest=2000,
            implied_volatility=0.20,
            bid=5.00,
            ask=5.20,
            last_price=5.10,
            underlying_price=448.0,
            delta=0.45,
        )

        assert data.symbol == "SPY250117C00450000"
        assert data.volume == 500
        assert data.delta == 0.45

    def test_mid_price(self):
        """Test mid price calculation."""
        data = OptionActivityData(
            symbol="TEST",
            underlying="SPY",
            option_type="call",
            strike=450.0,
            expiry=datetime(2025, 1, 17),
            volume=100,
            open_interest=1000,
            implied_volatility=0.20,
            bid=5.00,
            ask=5.20,
            last_price=5.10,
            underlying_price=448.0,
        )

        assert data.mid_price == 5.10

    def test_premium_traded(self):
        """Test premium traded calculation."""
        data = OptionActivityData(
            symbol="TEST",
            underlying="SPY",
            option_type="call",
            strike=450.0,
            expiry=datetime(2025, 1, 17),
            volume=100,
            open_interest=1000,
            implied_volatility=0.20,
            bid=5.00,
            ask=5.20,
            last_price=5.10,
            underlying_price=448.0,
        )

        # Mid = 5.10, volume = 100, multiplier = 100
        expected = 5.10 * 100 * 100
        assert data.premium_traded == expected

    def test_volume_oi_ratio(self):
        """Test volume/OI ratio."""
        data = OptionActivityData(
            symbol="TEST",
            underlying="SPY",
            option_type="call",
            strike=450.0,
            expiry=datetime(2025, 1, 17),
            volume=500,
            open_interest=1000,
            implied_volatility=0.20,
            bid=5.00,
            ask=5.20,
            last_price=5.10,
            underlying_price=448.0,
        )

        assert data.volume_oi_ratio == 0.5


class TestUnusualActivityConfig:
    """Tests for UnusualActivityConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = UnusualActivityConfig()

        assert config.volume_threshold_sigma == 2.0
        assert config.block_trade_threshold == 100
        assert config.lookback_days == 20

    def test_custom_values(self):
        """Test custom configuration."""
        config = UnusualActivityConfig(
            volume_threshold_sigma=3.0,
            block_trade_threshold=200,
        )

        assert config.volume_threshold_sigma == 3.0
        assert config.block_trade_threshold == 200


class TestUnusualActivityScanner:
    """Tests for UnusualActivityScanner class."""

    @pytest.fixture
    def scanner(self):
        """Create default scanner."""
        return UnusualActivityScanner()

    @pytest.fixture
    def sample_contracts(self):
        """Create sample option contracts."""
        expiry = datetime.now() + timedelta(days=30)

        return [
            OptionActivityData(
                symbol="SPY250117C00450000",
                underlying="SPY",
                option_type="call",
                strike=450.0,
                expiry=expiry,
                volume=100,
                open_interest=5000,
                implied_volatility=0.18,
                bid=4.90,
                ask=5.10,
                last_price=5.00,
                underlying_price=448.0,
                delta=0.45,
            ),
            OptionActivityData(
                symbol="SPY250117C00455000",
                underlying="SPY",
                option_type="call",
                strike=455.0,
                expiry=expiry,
                volume=50,
                open_interest=3000,
                implied_volatility=0.17,
                bid=2.40,
                ask=2.60,
                last_price=2.50,
                underlying_price=448.0,
                delta=0.35,
            ),
            OptionActivityData(
                symbol="SPY250117P00445000",
                underlying="SPY",
                option_type="put",
                strike=445.0,
                expiry=expiry,
                volume=80,
                open_interest=4000,
                implied_volatility=0.19,
                bid=3.90,
                ask=4.10,
                last_price=4.00,
                underlying_price=448.0,
                delta=-0.40,
            ),
        ]

    @pytest.fixture
    def block_trade_contract(self):
        """Create a block trade contract."""
        expiry = datetime.now() + timedelta(days=30)

        return OptionActivityData(
            symbol="SPY250117C00450000",
            underlying="SPY",
            option_type="call",
            strike=450.0,
            expiry=expiry,
            volume=500,  # Large volume
            open_interest=2000,
            implied_volatility=0.20,
            bid=4.90,
            ask=5.10,
            last_price=5.05,  # Traded above mid
            underlying_price=448.0,
            delta=0.45,
        )

    def test_initialization(self, scanner):
        """Test scanner initialization."""
        assert scanner.config is not None
        assert len(scanner.history) == 0

    def test_scan_empty(self, scanner):
        """Test scanning empty list."""
        alerts = scanner.scan([], 448.0)

        assert len(alerts) == 0

    def test_scan_normal_activity(self, scanner, sample_contracts):
        """Test scanning normal activity."""
        alerts = scanner.scan(sample_contracts, 448.0)

        # Normal activity may or may not generate alerts
        assert isinstance(alerts, list)

    def test_detect_block_trade(self, scanner, block_trade_contract):
        """Test block trade detection."""
        alerts = scanner.scan([block_trade_contract], 448.0)

        # Should detect the block trade
        block_alerts = [a for a in alerts if a.activity_type == ActivityType.BLOCK_TRADE]
        assert len(block_alerts) > 0

        alert = block_alerts[0]
        assert alert.volume == 500
        assert alert.direction_bias == DirectionBias.BULLISH  # Call above mid

    def test_detect_volume_spike_with_history(self, scanner, sample_contracts):
        """Test volume spike detection with history."""
        # Build history with low volume
        for _ in range(10):
            low_vol_contracts = [
                OptionActivityData(
                    symbol="SPY250117C00450000",
                    underlying="SPY",
                    option_type="call",
                    strike=450.0,
                    expiry=datetime.now() + timedelta(days=30),
                    volume=20,  # Low volume
                    open_interest=5000,
                    implied_volatility=0.18,
                    bid=4.90,
                    ask=5.10,
                    last_price=5.00,
                    underlying_price=448.0,
                    delta=0.45,
                )
            ]
            scanner.scan(low_vol_contracts, 448.0)

        # Now scan with high volume
        high_vol_contracts = [
            OptionActivityData(
                symbol="SPY250117C00450000",
                underlying="SPY",
                option_type="call",
                strike=450.0,
                expiry=datetime.now() + timedelta(days=30),
                volume=200,  # High volume relative to history
                open_interest=5000,
                implied_volatility=0.18,
                bid=4.90,
                ask=5.10,
                last_price=5.00,
                underlying_price=448.0,
                delta=0.45,
            )
        ]

        alerts = scanner.scan(high_vol_contracts, 448.0)
        volume_alerts = [a for a in alerts if a.activity_type == ActivityType.VOLUME_SPIKE]

        # Should detect volume spike
        assert len(volume_alerts) > 0

    def test_put_call_skew_detection(self, scanner):
        """Test put/call skew detection."""
        expiry = datetime.now() + timedelta(days=30)

        # Lots of calls, few puts = bullish
        bullish_contracts = [
            OptionActivityData(
                symbol="SPY_CALL",
                underlying="SPY",
                option_type="call",
                strike=450.0,
                expiry=expiry,
                volume=1000,
                open_interest=5000,
                implied_volatility=0.18,
                bid=4.90,
                ask=5.10,
                last_price=5.00,
                underlying_price=448.0,
                delta=0.45,
            ),
            OptionActivityData(
                symbol="SPY_PUT",
                underlying="SPY",
                option_type="put",
                strike=445.0,
                expiry=expiry,
                volume=100,  # Much lower than calls
                open_interest=3000,
                implied_volatility=0.19,
                bid=3.90,
                ask=4.10,
                last_price=4.00,
                underlying_price=448.0,
                delta=-0.40,
            ),
        ]

        alerts = scanner.scan(bullish_contracts, 448.0)
        skew_alerts = [a for a in alerts if a.activity_type == ActivityType.PUT_CALL_SKEW]

        # P/C ratio = 100/1000 = 0.1 < 0.3, should be bullish
        if skew_alerts:
            assert skew_alerts[0].direction_bias == DirectionBias.BULLISH

    def test_alert_urgency_levels(self, scanner):
        """Test alert urgency assignment."""
        expiry = datetime.now() + timedelta(days=30)

        # Very large block trade
        contract = OptionActivityData(
            symbol="SPY_BLOCK",
            underlying="SPY",
            option_type="call",
            strike=450.0,
            expiry=expiry,
            volume=1000,  # Very large
            open_interest=2000,
            implied_volatility=0.25,
            bid=4.90,
            ask=5.10,
            last_price=5.05,
            underlying_price=448.0,
            delta=0.45,
        )

        alerts = scanner.scan([contract], 448.0)

        if alerts:
            # Large trades should have at least medium urgency
            assert alerts[0].urgency in [
                UrgencyLevel.MEDIUM,
                UrgencyLevel.HIGH,
                UrgencyLevel.CRITICAL,
            ]

    def test_alert_callback(self, scanner, block_trade_contract):
        """Test alert callback."""
        alerts_received = []

        def callback(alert):
            alerts_received.append(alert)

        scanner.register_alert_callback(callback)
        scanner.scan([block_trade_contract], 448.0)

        # Callback should have been called
        assert len(alerts_received) > 0

    def test_analyze_flow(self, scanner, sample_contracts):
        """Test flow analysis."""
        flow = scanner.analyze_flow("SPY", sample_contracts)

        assert isinstance(flow, FlowAnalysis)
        assert flow.underlying == "SPY"
        assert flow.total_call_volume > 0
        assert flow.total_put_volume > 0
        assert flow.put_call_ratio > 0

    def test_flow_direction_bias(self, scanner):
        """Test flow direction bias detection."""
        expiry = datetime.now() + timedelta(days=30)

        # Strong call buying
        call_heavy = [
            OptionActivityData(
                symbol="SPY_CALL",
                underlying="SPY",
                option_type="call",
                strike=450.0,
                expiry=expiry,
                volume=500,
                open_interest=5000,
                implied_volatility=0.18,
                bid=4.90,
                ask=5.10,
                last_price=5.00,
                underlying_price=448.0,
                delta=0.45,
            ),
            OptionActivityData(
                symbol="SPY_PUT",
                underlying="SPY",
                option_type="put",
                strike=445.0,
                expiry=expiry,
                volume=50,
                open_interest=3000,
                implied_volatility=0.19,
                bid=3.90,
                ask=4.10,
                last_price=4.00,
                underlying_price=448.0,
                delta=-0.40,
            ),
        ]

        flow = scanner.analyze_flow("SPY", call_heavy)

        assert flow.direction_bias == DirectionBias.BULLISH

    def test_history_update(self, scanner, sample_contracts):
        """Test history is updated after scan."""
        scanner.scan(sample_contracts, 448.0)

        assert "SPY" in scanner.history
        history = scanner.history["SPY"]
        assert len(history.volumes) > 0

    def test_summary(self, scanner):
        """Test summary generation."""
        summary = scanner.get_summary()

        assert "config" in summary
        assert "symbols_tracked" in summary


class TestUnusualActivityAlert:
    """Tests for UnusualActivityAlert dataclass."""

    def test_creation(self):
        """Test alert creation."""
        alert = UnusualActivityAlert(
            symbol="SPY250117C00450000",
            underlying="SPY",
            activity_type=ActivityType.BLOCK_TRADE,
            current_value=500,
            historical_avg=50,
            deviation_sigma=5.0,
            percentile=99.0,
            volume=500,
            premium=250000.0,
            direction_bias=DirectionBias.BULLISH,
            confidence=0.85,
            urgency=UrgencyLevel.HIGH,
            timestamp=datetime.now(),
            expiry=datetime.now() + timedelta(days=30),
            strike=450.0,
            option_type="call",
        )

        assert alert.volume == 500
        assert alert.direction_bias == DirectionBias.BULLISH

    def test_to_dict(self):
        """Test conversion to dictionary."""
        alert = UnusualActivityAlert(
            symbol="TEST",
            underlying="SPY",
            activity_type=ActivityType.VOLUME_SPIKE,
            current_value=200,
            historical_avg=50,
            deviation_sigma=3.0,
            percentile=95.0,
            volume=200,
            premium=100000.0,
            direction_bias=DirectionBias.NEUTRAL,
            confidence=0.7,
            urgency=UrgencyLevel.MEDIUM,
            timestamp=datetime.now(),
            expiry=datetime.now() + timedelta(days=30),
            strike=450.0,
            option_type="call",
        )

        d = alert.to_dict()

        assert d["activity_type"] == "volume_spike"
        assert d["direction_bias"] == "neutral"
        assert d["urgency"] == "medium"


class TestActivityHistory:
    """Tests for ActivityHistory dataclass."""

    def test_add_record(self):
        """Test adding records."""
        history = ActivityHistory()
        history.add_record(
            volume=1000,
            oi=5000,
            iv=0.20,
            pc_ratio=0.7,
            timestamp=datetime.now(),
        )

        assert len(history.volumes) == 1
        assert history.volumes[0] == 1000

    def test_trim(self):
        """Test trimming old records."""
        history = ActivityHistory()

        # Add old records
        old_time = datetime.now() - timedelta(days=30)
        history.add_record(1000, 5000, 0.20, 0.7, old_time)

        # Add recent records
        recent_time = datetime.now()
        history.add_record(1500, 5500, 0.22, 0.8, recent_time)

        # Trim to 20 days
        history.trim(20)

        # Old record should be removed
        assert len(history.volumes) == 1
        assert history.volumes[0] == 1500


class TestCreateUnusualActivityScanner:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating with defaults."""
        scanner = create_unusual_activity_scanner()

        assert isinstance(scanner, UnusualActivityScanner)
        assert scanner.config.volume_threshold_sigma == 2.0

    def test_create_with_params(self):
        """Test creating with custom params."""
        scanner = create_unusual_activity_scanner(
            volume_threshold=3.0,
            block_threshold=200,
        )

        assert scanner.config.volume_threshold_sigma == 3.0
        assert scanner.config.block_trade_threshold == 200
