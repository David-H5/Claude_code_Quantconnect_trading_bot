"""
Volatility Bands Indicator

A custom indicator that creates dynamic bands around price based on
Average True Range (ATR) to identify potential support/resistance levels.

This indicator is useful for:
- Identifying overbought/oversold conditions
- Setting dynamic stop-loss levels
- Detecting volatility expansion/contraction

Author: QuantConnect Trading Bot
Date: 2025-11-25
"""

import logging


logger = logging.getLogger(__name__)


class VolatilityBands:
    """
    Dynamic volatility bands using ATR-based calculations.

    The indicator creates upper and lower bands around a moving average,
    with band width determined by a multiple of the Average True Range.

    Attributes:
        period: Lookback period for calculations
        multiplier: ATR multiplier for band width
        upper_band: Current upper band value
        middle_band: Current middle band (SMA) value
        lower_band: Current lower band value
        atr: Current ATR value
        is_ready: Whether indicator has enough data
    """

    def __init__(self, period: int = 20, multiplier: float = 2.0):
        """
        Initialize VolatilityBands indicator.

        Args:
            period: Lookback period for SMA and ATR calculations
            multiplier: ATR multiplier for band width (default 2.0)
        """
        self.period = period
        self.multiplier = multiplier

        # Data storage
        self._prices: list[float] = []
        self._true_ranges: list[float] = []
        self._prev_close: float | None = None

        # Output values
        self.upper_band: float = 0.0
        self.middle_band: float = 0.0
        self.lower_band: float = 0.0
        self.atr: float = 0.0
        self.band_width: float = 0.0

        # State
        self._samples = 0

    @property
    def is_ready(self) -> bool:
        """Check if indicator has enough data to produce valid values."""
        return self._samples >= self.period

    @property
    def IsReady(self) -> bool:
        """QuantConnect-style property for compatibility."""
        return self.is_ready

    def update(self, high: float, low: float, close: float) -> bool:
        """
        Update the indicator with new price data.

        Args:
            high: High price of the bar
            low: Low price of the bar
            close: Close price of the bar

        Returns:
            True if indicator is ready, False otherwise
        """
        self._samples += 1

        # Calculate True Range
        if self._prev_close is not None:
            true_range = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
        else:
            true_range = high - low

        self._true_ranges.append(true_range)
        self._prices.append(close)

        # Keep only the data we need
        if len(self._prices) > self.period:
            self._prices.pop(0)
            self._true_ranges.pop(0)

        self._prev_close = close

        # Calculate indicator values if ready
        if self.is_ready:
            self._calculate()

        return self.is_ready

    def Update(self, high: float, low: float, close: float) -> bool:
        """QuantConnect-style method for compatibility."""
        return self.update(high, low, close)

    def _calculate(self) -> None:
        """Calculate band values from current data."""
        # Simple Moving Average
        self.middle_band = sum(self._prices) / len(self._prices)

        # Average True Range
        self.atr = sum(self._true_ranges) / len(self._true_ranges)

        # Calculate bands
        band_offset = self.atr * self.multiplier
        self.upper_band = self.middle_band + band_offset
        self.lower_band = self.middle_band - band_offset

        # Band width as percentage of middle band
        if self.middle_band > 0:
            self.band_width = (self.upper_band - self.lower_band) / self.middle_band

    def get_position(self, price: float) -> float:
        """
        Get the relative position of price within the bands.

        Args:
            price: Current price to evaluate

        Returns:
            Float between 0 and 1, where:
            - 0 = at or below lower band
            - 0.5 = at middle band
            - 1 = at or above upper band
        """
        if not self.is_ready or self.upper_band == self.lower_band:
            return 0.5

        position = (price - self.lower_band) / (self.upper_band - self.lower_band)
        return max(0.0, min(1.0, position))

    def is_overbought(self, price: float, threshold: float = 0.9) -> bool:
        """
        Check if price is in overbought territory.

        Args:
            price: Current price
            threshold: Position threshold (default 0.9 = 90% toward upper band)

        Returns:
            True if price is above threshold position
        """
        return self.is_ready and self.get_position(price) >= threshold

    def is_oversold(self, price: float, threshold: float = 0.1) -> bool:
        """
        Check if price is in oversold territory.

        Args:
            price: Current price
            threshold: Position threshold (default 0.1 = 10% from lower band)

        Returns:
            True if price is below threshold position
        """
        return self.is_ready and self.get_position(price) <= threshold

    def reset(self) -> None:
        """Reset the indicator to initial state."""
        self._prices.clear()
        self._true_ranges.clear()
        self._prev_close = None
        self._samples = 0
        self.upper_band = 0.0
        self.middle_band = 0.0
        self.lower_band = 0.0
        self.atr = 0.0
        self.band_width = 0.0


class KeltnerChannels(VolatilityBands):
    """
    Keltner Channels - a specific implementation of volatility bands
    using EMA instead of SMA for the middle band.

    Keltner Channels are commonly used for trend-following strategies
    and breakout detection.
    """

    def __init__(self, period: int = 20, multiplier: float = 2.0):
        """
        Initialize Keltner Channels.

        Args:
            period: Lookback period (default 20)
            multiplier: ATR multiplier (default 2.0)
        """
        super().__init__(period, multiplier)
        self._ema_multiplier = 2 / (period + 1)
        self._ema: float | None = None

    def update(self, high: float, low: float, close: float) -> bool:
        """
        Update indicator with new price data, calculating EMA progressively.

        Args:
            high: High price of the bar
            low: Low price of the bar
            close: Close price of the bar

        Returns:
            True if indicator is ready, False otherwise
        """
        # Update EMA progressively on each new price
        if self._ema is None:
            self._ema = close
        else:
            self._ema = (close * self._ema_multiplier) + (self._ema * (1 - self._ema_multiplier))

        # Call parent update for price/TR tracking
        return super().update(high, low, close)

    def _calculate(self) -> None:
        """Calculate Keltner Channel values using pre-computed EMA."""
        # Use the progressively calculated EMA
        self.middle_band = self._ema

        # ATR calculation (same as parent)
        self.atr = sum(self._true_ranges) / len(self._true_ranges)

        # Calculate bands
        band_offset = self.atr * self.multiplier
        self.upper_band = self.middle_band + band_offset
        self.lower_band = self.middle_band - band_offset

        if self.middle_band > 0:
            self.band_width = (self.upper_band - self.lower_band) / self.middle_band

    def reset(self) -> None:
        """Reset indicator including EMA state."""
        super().reset()
        self._ema = None
