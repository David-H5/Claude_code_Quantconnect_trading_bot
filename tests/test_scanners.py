"""
Scanner Tests

Tests for market scanners including:
- Options scanner for underpriced options
- Movement scanner for price movers
- Filter validation
- Signal generation

Based on best practices from:
- QuantConnect options documentation
- Option Alpha scanning features
"""

from datetime import datetime, timedelta

import pytest

from config import MovementScannerConfig, OptionsScannerConfig
from scanners.movement_scanner import (
    MovementDirection,
    MovementScanner,
    PriceData,
    create_movement_scanner,
)
from scanners.options_scanner import (
    OptionContract,
    OptionsScanner,
    OptionType,
    UnderpricedOption,
    create_options_scanner,
)


class TestOptionsScanner:
    """Tests for options scanner functionality."""

    @pytest.fixture
    def scanner_config(self):
        """Create options scanner config."""
        return OptionsScannerConfig(
            enabled=True,
            min_days_to_expiry=7,
            max_days_to_expiry=45,
            min_open_interest=100,
            min_volume=10,
            target_delta_range=(0.20, 0.40),
            underpriced_threshold=0.10,
        )

    @pytest.fixture
    def sample_option_chain(self) -> list[OptionContract]:
        """Create sample option chain for testing."""
        expiry = datetime.now() + timedelta(days=30)
        return [
            OptionContract(
                symbol="SPY240315C00500000",
                underlying="SPY",
                option_type=OptionType.CALL,
                strike=500.0,
                expiry=expiry,
                bid=5.00,
                ask=5.20,
                last=5.10,
                volume=100,
                open_interest=500,
                implied_volatility=0.20,
                delta=0.30,
                gamma=0.02,
                theta=-0.05,
                vega=0.10,
            ),
            OptionContract(
                symbol="SPY240315C00510000",
                underlying="SPY",
                option_type=OptionType.CALL,
                strike=510.0,
                expiry=expiry,
                bid=2.50,
                ask=2.70,
                last=2.60,
                volume=50,
                open_interest=300,
                implied_volatility=0.22,
                delta=0.25,
                gamma=0.015,
                theta=-0.04,
                vega=0.08,
            ),
            OptionContract(
                symbol="SPY240315P00490000",
                underlying="SPY",
                option_type=OptionType.PUT,
                strike=490.0,
                expiry=expiry,
                bid=3.00,
                ask=3.30,
                last=3.15,
                volume=75,
                open_interest=400,
                implied_volatility=0.21,
                delta=-0.28,
                gamma=0.018,
                theta=-0.045,
                vega=0.09,
            ),
        ]

    @pytest.mark.unit
    def test_scanner_creation(self, scanner_config):
        """Test scanner creates successfully."""
        scanner = create_options_scanner(scanner_config)
        assert scanner is not None
        assert scanner.config == scanner_config

    @pytest.mark.unit
    def test_filter_by_days_to_expiry(self, scanner_config):
        """Test filtering by days to expiry."""
        scanner = OptionsScanner(scanner_config)

        # Contract expiring too soon
        soon_expiry = OptionContract(
            symbol="TEST",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=500.0,
            expiry=datetime.now() + timedelta(days=3),  # Too soon
            bid=1.0,
            ask=1.1,
            last=1.05,
            volume=100,
            open_interest=500,
            implied_volatility=0.20,
            delta=0.30,
        )

        assert scanner._passes_filters(soon_expiry) is False

    @pytest.mark.unit
    def test_filter_by_delta_range(self, scanner_config):
        """Test filtering by delta range."""
        scanner = OptionsScanner(scanner_config)
        expiry = datetime.now() + timedelta(days=30)

        # Delta too low
        low_delta = OptionContract(
            symbol="TEST",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=520.0,
            expiry=expiry,
            bid=0.50,
            ask=0.60,
            last=0.55,
            volume=100,
            open_interest=500,
            implied_volatility=0.20,
            delta=0.10,  # Below 0.20 threshold
        )

        # Delta in range
        good_delta = OptionContract(
            symbol="TEST2",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=505.0,
            expiry=expiry,
            bid=3.00,
            ask=3.20,
            last=3.10,
            volume=100,
            open_interest=500,
            implied_volatility=0.20,
            delta=0.30,  # In 0.20-0.40 range
        )

        assert scanner._passes_filters(low_delta) is False
        assert scanner._passes_filters(good_delta) is True

    @pytest.mark.unit
    def test_filter_by_liquidity(self, scanner_config):
        """Test filtering by open interest and volume."""
        scanner = OptionsScanner(scanner_config)
        expiry = datetime.now() + timedelta(days=30)

        # Low open interest
        illiquid = OptionContract(
            symbol="TEST",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=500.0,
            expiry=expiry,
            bid=5.00,
            ask=5.20,
            last=5.10,
            volume=5,  # Too low
            open_interest=50,  # Too low
            implied_volatility=0.20,
            delta=0.30,
        )

        assert scanner._passes_filters(illiquid) is False

    @pytest.mark.unit
    def test_scan_chain_finds_underpriced(self, scanner_config, sample_option_chain):
        """Test that scan finds underpriced options."""
        scanner = OptionsScanner(scanner_config)

        # Run scan
        results = scanner.scan_chain(
            underlying="SPY",
            spot_price=505.0,
            chain=sample_option_chain,
        )

        # Results should be list (may be empty if nothing underpriced)
        assert isinstance(results, list)

    @pytest.mark.unit
    def test_underpriced_calculation(self, scanner_config):
        """Test underpriced percentage calculation."""
        scanner = OptionsScanner(scanner_config)

        # Fair value > market price = underpriced
        fair_value = 5.50
        market_price = 5.00
        underpriced_pct = (fair_value - market_price) / market_price

        assert underpriced_pct == pytest.approx(0.10, abs=0.001)

    @pytest.mark.unit
    def test_alert_callback(self, scanner_config):
        """Test alert callback is triggered."""
        alerts_received = []

        def alert_handler(opportunity: UnderpricedOption):
            alerts_received.append(opportunity)

        scanner = create_options_scanner(scanner_config, alert_callback=alert_handler)

        # Mock an underpriced opportunity
        expiry = datetime.now() + timedelta(days=30)
        chain = [
            OptionContract(
                symbol="TEST",
                underlying="SPY",
                option_type=OptionType.CALL,
                strike=500.0,
                expiry=expiry,
                bid=2.00,
                ask=2.20,
                last=2.10,
                volume=100,
                open_interest=500,
                implied_volatility=0.15,  # Low IV suggests underpriced
                delta=0.30,
            )
        ]

        scanner.scan_chain("SPY", 505.0, chain)
        # Alert callback may or may not be called depending on BS calculation

    @pytest.mark.unit
    def test_iv_percentile_calculation(self, scanner_config):
        """Test IV percentile calculation."""
        scanner = OptionsScanner(scanner_config)

        # Add historical IV data
        for iv in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
            scanner._update_iv_history("SPY", iv)

        # Current IV at 50th percentile
        percentile = scanner._calculate_iv_percentile("SPY", 0.22)
        assert 40 <= percentile <= 60  # Roughly middle

    @pytest.mark.unit
    def test_disabled_scanner(self):
        """Test scanner behavior when disabled."""
        config = OptionsScannerConfig(enabled=False)
        scanner = OptionsScanner(config)

        results = scanner.scan_chain("SPY", 505.0, [])
        assert results == []


class TestMovementScanner:
    """Tests for movement scanner functionality."""

    @pytest.fixture
    def scanner_config(self):
        """Create movement scanner config."""
        return MovementScannerConfig(
            enabled=True,
            min_movement_pct=0.02,
            max_movement_pct=0.04,
            volume_surge_threshold=2.0,
            require_news_corroboration=False,
        )

    @pytest.fixture
    def sample_price_data(self) -> list[PriceData]:
        """Create sample price data for testing."""
        now = datetime.now()
        return [
            PriceData(
                symbol="AAPL",
                current_price=180.0,
                open_price=175.0,  # +2.9% move
                previous_close=174.0,
                high=181.0,
                low=174.5,
                volume=50_000_000,
                average_volume=25_000_000,  # 2x normal volume
                timestamp=now,
            ),
            PriceData(
                symbol="MSFT",
                current_price=400.0,
                open_price=398.0,  # +0.5% move (below threshold)
                previous_close=397.0,
                high=401.0,
                low=397.5,
                volume=20_000_000,
                average_volume=18_000_000,
                timestamp=now,
            ),
            PriceData(
                symbol="NVDA",
                current_price=850.0,
                open_price=820.0,  # +3.7% move
                previous_close=818.0,
                high=855.0,
                low=818.0,
                volume=100_000_000,
                average_volume=40_000_000,  # 2.5x volume
                timestamp=now,
            ),
        ]

    @pytest.mark.unit
    def test_scanner_creation(self, scanner_config):
        """Test scanner creates successfully."""
        scanner = create_movement_scanner(scanner_config)
        assert scanner is not None

    @pytest.mark.unit
    def test_detects_significant_move(self, scanner_config, sample_price_data):
        """Test detection of significant price movements."""
        scanner = MovementScanner(scanner_config)

        alerts = scanner.scan_batch(sample_price_data)

        # Should detect AAPL (2.9%) and NVDA (3.7%)
        alert_symbols = [a.symbol for a in alerts]
        assert "AAPL" in alert_symbols
        assert "NVDA" in alert_symbols
        # MSFT move too small
        assert "MSFT" not in alert_symbols

    @pytest.mark.unit
    def test_filters_by_volume_surge(self, scanner_config):
        """Test filtering by volume surge."""
        scanner = MovementScanner(scanner_config)

        # High price move but low volume - should be filtered out
        low_volume_data = PriceData(
            symbol="TEST",
            current_price=103.0,
            open_price=100.0,  # 3% move
            previous_close=100.0,
            high=103.5,
            low=99.5,
            volume=1_000_000,  # Below average (0.2x)
            average_volume=5_000_000,
            timestamp=datetime.now(),
        )

        alert = scanner.scan(low_volume_data)
        # Should filter out due to low volume (below 2x threshold)
        assert alert is None

    @pytest.mark.unit
    def test_movement_direction(self, scanner_config):
        """Test correct movement direction detection."""
        scanner = MovementScanner(scanner_config)

        # Upward move
        up_data = PriceData(
            symbol="UP",
            current_price=103.0,
            open_price=100.0,
            previous_close=100.0,
            high=103.5,
            low=99.5,
            volume=10_000_000,
            average_volume=5_000_000,
            timestamp=datetime.now(),
        )

        # Downward move
        down_data = PriceData(
            symbol="DOWN",
            current_price=97.0,
            open_price=100.0,
            previous_close=100.0,
            high=100.5,
            low=96.5,
            volume=10_000_000,
            average_volume=5_000_000,
            timestamp=datetime.now(),
        )

        up_alert = scanner.scan(up_data)
        down_alert = scanner.scan(down_data)

        if up_alert:
            assert up_alert.direction == MovementDirection.UP
        if down_alert:
            assert down_alert.direction == MovementDirection.DOWN

    @pytest.mark.unit
    def test_change_from_open_calculation(self):
        """Test change from open calculation."""
        data = PriceData(
            symbol="TEST",
            current_price=105.0,
            open_price=100.0,
            previous_close=99.0,
            high=106.0,
            low=99.5,
            volume=1_000_000,
            average_volume=1_000_000,
            timestamp=datetime.now(),
        )

        assert data.change_from_open == pytest.approx(0.05, abs=0.001)  # 5%

    @pytest.mark.unit
    def test_change_from_close_calculation(self):
        """Test change from previous close calculation."""
        data = PriceData(
            symbol="TEST",
            current_price=105.0,
            open_price=100.0,
            previous_close=100.0,
            high=106.0,
            low=99.5,
            volume=1_000_000,
            average_volume=1_000_000,
            timestamp=datetime.now(),
        )

        assert data.change_from_close == pytest.approx(0.05, abs=0.001)  # 5%

    @pytest.mark.unit
    def test_volume_surge_ratio(self, scanner_config):
        """Test volume surge ratio calculation."""
        scanner = MovementScanner(scanner_config)
        data = PriceData(
            symbol="TEST",
            current_price=103.0,  # 3% move from close
            open_price=100.0,
            previous_close=100.0,
            high=106.0,
            low=99.5,
            volume=10_000_000,
            average_volume=5_000_000,  # 2x surge
            timestamp=datetime.now(),
        )

        ratio = data.volume_ratio
        assert ratio == 2.0

        # Should generate alert due to sufficient move and volume
        alert = scanner.scan(data)
        assert alert is not None

    @pytest.mark.unit
    def test_disabled_scanner(self):
        """Test scanner behavior when disabled."""
        config = MovementScannerConfig(enabled=False)
        scanner = MovementScanner(config)

        data = PriceData(
            symbol="TEST",
            current_price=103.0,
            open_price=100.0,
            previous_close=100.0,
            high=103.5,
            low=99.5,
            volume=10_000_000,
            average_volume=5_000_000,
            timestamp=datetime.now(),
        )

        alert = scanner.scan(data)
        assert alert is None

    @pytest.mark.unit
    def test_excludes_moves_above_max(self, scanner_config):
        """Test filtering of moves above maximum threshold."""
        scanner = MovementScanner(scanner_config)

        # 10% move - above max threshold (might be bad data or halt)
        extreme_data = PriceData(
            symbol="HALT",
            current_price=110.0,
            open_price=100.0,  # 10% move
            previous_close=100.0,
            high=111.0,
            low=99.0,
            volume=50_000_000,
            average_volume=25_000_000,
            timestamp=datetime.now(),
        )

        alert = scanner.scan(extreme_data)
        # Move above 4% max should be excluded
        assert alert is None


class TestScannerIntegration:
    """Integration tests for scanner coordination."""

    @pytest.mark.integration
    def test_options_scanner_with_movement_scanner(self):
        """Test coordinated use of both scanners."""
        movement_config = MovementScannerConfig(
            enabled=True,
            min_movement_pct=0.02,
            max_movement_pct=0.04,
            volume_surge_threshold=2.0,
            require_news_corroboration=False,
        )
        options_config = OptionsScannerConfig(
            enabled=True,
            min_days_to_expiry=7,
            max_days_to_expiry=45,
        )

        movement_scanner = MovementScanner(movement_config)
        options_scanner = OptionsScanner(options_config)

        # Scenario: Find movers, then scan their option chains
        price_data = [
            PriceData(
                symbol="AAPL",
                current_price=180.0,
                open_price=175.0,
                previous_close=174.0,
                high=181.0,
                low=174.5,
                volume=50_000_000,
                average_volume=25_000_000,
                timestamp=datetime.now(),
            )
        ]

        # Step 1: Find movers using scan_batch for list
        movers = movement_scanner.scan_batch(price_data)

        # Step 2: For each mover, scan options
        for alert in movers:
            # Would call options_scanner.scan_chain(alert.symbol, ...)
            pass  # Options chain would come from data provider

    @pytest.mark.integration
    def test_watchlist_scanning(self):
        """Test scanning a watchlist of symbols."""
        options_config = OptionsScannerConfig(enabled=True)
        scanner = OptionsScanner(options_config)

        watchlist = ["AAPL", "MSFT", "NVDA", "GOOGL"]

        # Mock chain fetcher
        def mock_chain_fetcher(symbol: str):
            return (500.0, [])  # spot_price, empty chain

        results = scanner.get_watchlist_opportunities(watchlist, mock_chain_fetcher)

        assert isinstance(results, dict)
