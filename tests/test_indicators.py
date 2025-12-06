"""
Tests for custom technical indicators.
"""

import pytest

from indicators.volatility_bands import KeltnerChannels, VolatilityBands


class TestVolatilityBands:
    """Test suite for VolatilityBands indicator."""

    @pytest.fixture
    def indicator(self):
        """Create a default VolatilityBands indicator."""
        return VolatilityBands(period=20, multiplier=2.0)

    @pytest.fixture
    def small_indicator(self):
        """Create a small period indicator for testing."""
        return VolatilityBands(period=3, multiplier=2.0)

    @pytest.mark.unit
    def test_initialization(self, indicator):
        """Test indicator initializes with correct defaults."""
        assert indicator.period == 20
        assert indicator.multiplier == 2.0
        assert indicator.is_ready is False
        assert indicator.upper_band == 0.0
        assert indicator.middle_band == 0.0
        assert indicator.lower_band == 0.0

    @pytest.mark.unit
    def test_not_ready_before_period(self, small_indicator):
        """Test indicator is not ready before enough samples."""
        ind = small_indicator

        # Add 2 samples (need 3)
        ind.update(high=101, low=99, close=100)
        assert ind.is_ready is False

        ind.update(high=102, low=98, close=101)
        assert ind.is_ready is False

    @pytest.mark.unit
    def test_ready_after_period(self, small_indicator):
        """Test indicator becomes ready after period samples."""
        ind = small_indicator

        ind.update(high=101, low=99, close=100)
        ind.update(high=102, low=98, close=101)
        ind.update(high=103, low=97, close=102)

        assert ind.is_ready is True
        assert ind.middle_band > 0
        assert ind.upper_band > ind.middle_band
        assert ind.lower_band < ind.middle_band

    @pytest.mark.unit
    def test_band_calculation(self, small_indicator):
        """Test band values are calculated correctly."""
        ind = small_indicator

        # Add consistent data
        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)

        # Middle band should be SMA
        assert ind.middle_band == 100.0

        # ATR should be 4 (high - low with no gaps)
        assert ind.atr == 4.0

        # Bands should be middle +/- (ATR * multiplier)
        assert ind.upper_band == 108.0  # 100 + (4 * 2)
        assert ind.lower_band == 92.0  # 100 - (4 * 2)

    @pytest.mark.unit
    def test_true_range_with_gaps(self):
        """Test True Range calculation with price gaps."""
        ind = VolatilityBands(period=2, multiplier=1.0)

        # First bar: TR = high - low = 2
        ind.update(high=101, low=99, close=100)

        # Second bar with gap up: TR should use prev close
        # high=110, low=108, prev_close=100
        # TR = max(110-108, |110-100|, |108-100|) = max(2, 10, 8) = 10
        ind.update(high=110, low=108, close=109)

        assert ind.is_ready
        # ATR = (2 + 10) / 2 = 6
        assert ind.atr == 6.0

    @pytest.mark.unit
    def test_get_position_at_bands(self, small_indicator):
        """Test position calculation at band levels."""
        ind = small_indicator

        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)

        # At lower band (92)
        assert ind.get_position(92.0) == pytest.approx(0.0)

        # At middle band (100)
        assert ind.get_position(100.0) == pytest.approx(0.5)

        # At upper band (108)
        assert ind.get_position(108.0) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_get_position_clamped(self, small_indicator):
        """Test position is clamped between 0 and 1."""
        ind = small_indicator

        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)

        # Below lower band
        assert ind.get_position(80.0) == 0.0

        # Above upper band
        assert ind.get_position(120.0) == 1.0

    @pytest.mark.unit
    def test_is_overbought(self, small_indicator):
        """Test overbought detection."""
        ind = small_indicator

        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)

        # Upper band is 108
        assert ind.is_overbought(108.0) is True
        assert ind.is_overbought(107.5) is True  # 0.97 position
        assert ind.is_overbought(100.0) is False  # 0.5 position

    @pytest.mark.unit
    def test_is_oversold(self, small_indicator):
        """Test oversold detection."""
        ind = small_indicator

        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)

        # Lower band is 92
        assert ind.is_oversold(92.0) is True
        assert ind.is_oversold(93.0) is True  # 0.06 position
        assert ind.is_oversold(100.0) is False  # 0.5 position

    @pytest.mark.unit
    def test_reset(self, small_indicator):
        """Test indicator reset."""
        ind = small_indicator

        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)

        assert ind.is_ready is True

        ind.reset()

        assert ind.is_ready is False
        assert ind.upper_band == 0.0
        assert ind.middle_band == 0.0
        assert ind.lower_band == 0.0
        assert ind.atr == 0.0

    @pytest.mark.unit
    def test_quantconnect_compatibility(self, indicator):
        """Test QuantConnect-style API compatibility."""
        ind = indicator

        # Should have IsReady property
        assert hasattr(ind, "IsReady")
        assert ind.IsReady is False

        # Should have Update method
        assert hasattr(ind, "Update")


class TestKeltnerChannels:
    """Test suite for KeltnerChannels indicator."""

    @pytest.fixture
    def indicator(self):
        """Create a KeltnerChannels indicator."""
        return KeltnerChannels(period=3, multiplier=2.0)

    @pytest.mark.unit
    def test_ema_middle_band(self, indicator):
        """Test that Keltner uses EMA for middle band."""
        ind = indicator

        # Add data
        ind.update(high=102, low=98, close=100)
        ind.update(high=104, low=96, close=105)
        ind.update(high=108, low=102, close=110)

        # EMA should weight recent prices more than SMA
        sma = (100 + 105 + 110) / 3  # 105
        assert ind.middle_band != sma  # Should be different from SMA
        assert ind.middle_band > sma  # EMA should be higher due to uptrend

    @pytest.mark.unit
    def test_inherits_volatility_bands(self, indicator):
        """Test KeltnerChannels inherits from VolatilityBands."""
        assert isinstance(indicator, VolatilityBands)
        assert hasattr(indicator, "is_overbought")
        assert hasattr(indicator, "is_oversold")
        assert hasattr(indicator, "get_position")

    @pytest.mark.unit
    def test_reset_clears_ema(self, indicator):
        """Test reset clears EMA state."""
        ind = indicator

        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)

        ind.reset()

        assert ind._ema is None
        assert ind.is_ready is False


class TestIndicatorEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_zero_volatility(self):
        """Test handling of zero volatility (constant price)."""
        ind = VolatilityBands(period=3, multiplier=2.0)

        # Same price every bar
        ind.update(high=100, low=100, close=100)
        ind.update(high=100, low=100, close=100)
        ind.update(high=100, low=100, close=100)

        assert ind.atr == 0.0
        assert ind.upper_band == ind.middle_band
        assert ind.lower_band == ind.middle_band

    @pytest.mark.unit
    def test_large_gap(self):
        """Test handling of large price gaps."""
        ind = VolatilityBands(period=2, multiplier=1.0)

        ind.update(high=100, low=99, close=100)
        ind.update(high=200, low=199, close=200)  # Large gap up

        assert ind.is_ready
        # TR should capture the gap
        assert ind.atr > 50

    @pytest.mark.unit
    def test_band_width_percentage(self):
        """Test band width as percentage calculation."""
        ind = VolatilityBands(period=3, multiplier=2.0)

        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)
        ind.update(high=102, low=98, close=100)

        # Band width = (108 - 92) / 100 = 0.16
        assert ind.band_width == pytest.approx(0.16)
