"""
Technical Analysis Alpha Models

Provides alpha signals from various technical indicators:
- VWAP (Volume Weighted Average Price)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- CCI (Commodity Channel Index)
- Bollinger Bands and BBW
- OBV (On-Balance Volume)
- Ichimoku Cloud
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal enumeration."""

    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class AlphaSignal:
    """Alpha signal from a technical indicator."""

    indicator: str
    signal: Signal
    strength: float  # 0.0 to 1.0
    value: float  # Current indicator value
    threshold: float | None = None
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "indicator": self.indicator,
            "signal": self.signal.name,
            "strength": self.strength,
            "value": self.value,
            "threshold": self.threshold,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


class VWAPIndicator:
    """
    Volume Weighted Average Price indicator.

    VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
    """

    def __init__(self, anchor: str = "session"):
        """
        Initialize VWAP indicator.

        Args:
            anchor: "session" for daily reset, "week", "month"
        """
        self.anchor = anchor
        self.reset()

    def reset(self) -> None:
        """Reset cumulative values."""
        self._cumulative_pv = 0.0
        self._cumulative_volume = 0.0
        self._last_date: datetime | None = None

    def update(
        self,
        price: float,
        volume: int,
        timestamp: datetime | None = None,
    ) -> float:
        """
        Update VWAP with new data.

        Args:
            price: Typical price (H+L+C)/3 or trade price
            volume: Volume
            timestamp: Timestamp for anchor checking

        Returns:
            Current VWAP value
        """
        timestamp = timestamp or datetime.now()

        # Check for anchor reset
        if self._should_reset(timestamp):
            self.reset()

        self._cumulative_pv += price * volume
        self._cumulative_volume += volume
        self._last_date = timestamp

        if self._cumulative_volume > 0:
            return self._cumulative_pv / self._cumulative_volume
        return price

    def _should_reset(self, timestamp: datetime) -> bool:
        """Check if VWAP should reset based on anchor."""
        if self._last_date is None:
            return True

        if self.anchor == "session":
            return timestamp.date() != self._last_date.date()
        elif self.anchor == "week":
            return timestamp.isocalendar()[1] != self._last_date.isocalendar()[1]
        elif self.anchor == "month":
            return timestamp.month != self._last_date.month

        return False

    def generate_signal(self, current_price: float, vwap: float) -> AlphaSignal:
        """
        Generate trading signal from VWAP.

        Args:
            current_price: Current market price
            vwap: Current VWAP value

        Returns:
            AlphaSignal
        """
        deviation = (current_price - vwap) / vwap if vwap > 0 else 0

        if deviation > 0.02:
            signal = Signal.SELL
            reason = f"Price {deviation:.1%} above VWAP - overbought"
        elif deviation < -0.02:
            signal = Signal.BUY
            reason = f"Price {deviation:.1%} below VWAP - oversold"
        else:
            signal = Signal.NEUTRAL
            reason = "Price near VWAP"

        return AlphaSignal(
            indicator="VWAP",
            signal=signal,
            strength=min(abs(deviation) * 10, 1.0),
            value=vwap,
            reason=reason,
        )


class RSIIndicator:
    """
    Relative Strength Index indicator.

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """

    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        """
        Initialize RSI indicator.

        Args:
            period: RSI period
            overbought: Overbought threshold
            oversold: Oversold threshold
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self._prices: list[float] = []
        self._avg_gain = 0.0
        self._avg_loss = 0.0

    @property
    def is_ready(self) -> bool:
        """Check if indicator has enough data."""
        return len(self._prices) >= self.period + 1

    def update(self, price: float) -> float:
        """
        Update RSI with new price.

        Args:
            price: Close price

        Returns:
            Current RSI value
        """
        self._prices.append(price)

        if len(self._prices) < 2:
            return 50.0

        # Calculate change
        change = self._prices[-1] - self._prices[-2]
        gain = max(change, 0)
        loss = abs(min(change, 0))

        if len(self._prices) == self.period + 1:
            # Initial calculation
            gains = []
            losses = []
            for i in range(1, len(self._prices)):
                c = self._prices[i] - self._prices[i - 1]
                gains.append(max(c, 0))
                losses.append(abs(min(c, 0)))
            self._avg_gain = sum(gains) / self.period
            self._avg_loss = sum(losses) / self.period
        elif len(self._prices) > self.period + 1:
            # Smoothed calculation
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        # Keep only necessary prices
        if len(self._prices) > self.period + 2:
            self._prices = self._prices[-self.period - 2 :]

        # Calculate RSI
        if self._avg_loss == 0:
            return 100.0

        rs = self._avg_gain / self._avg_loss
        return 100 - (100 / (1 + rs))

    def generate_signal(self, rsi: float) -> AlphaSignal:
        """
        Generate trading signal from RSI.

        Args:
            rsi: Current RSI value

        Returns:
            AlphaSignal
        """
        if rsi >= self.overbought:
            signal = Signal.SELL if rsi < 80 else Signal.STRONG_SELL
            reason = f"RSI {rsi:.1f} - overbought"
            strength = (rsi - self.overbought) / (100 - self.overbought)
        elif rsi <= self.oversold:
            signal = Signal.BUY if rsi > 20 else Signal.STRONG_BUY
            reason = f"RSI {rsi:.1f} - oversold"
            strength = (self.oversold - rsi) / self.oversold
        else:
            signal = Signal.NEUTRAL
            reason = f"RSI {rsi:.1f} - neutral"
            strength = 0.0

        return AlphaSignal(
            indicator="RSI",
            signal=signal,
            strength=min(strength, 1.0),
            value=rsi,
            threshold=self.overbought if rsi > 50 else self.oversold,
            reason=reason,
        )


class MACDIndicator:
    """
    Moving Average Convergence Divergence indicator.

    MACD Line = Fast EMA - Slow EMA
    Signal Line = EMA of MACD Line
    Histogram = MACD Line - Signal Line
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Initialize MACD indicator.

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        """
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self._fast_ema = 0.0
        self._slow_ema = 0.0
        self._signal_ema = 0.0
        self._prices: list[float] = []
        self._macd_values: list[float] = []

    @property
    def is_ready(self) -> bool:
        """Check if indicator has enough data."""
        return len(self._prices) >= self.slow + self.signal

    def _ema_multiplier(self, period: int) -> float:
        """Calculate EMA multiplier."""
        return 2 / (period + 1)

    def update(self, price: float) -> tuple[float, float, float]:
        """
        Update MACD with new price.

        Args:
            price: Close price

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        self._prices.append(price)

        if len(self._prices) == 1:
            self._fast_ema = price
            self._slow_ema = price
        else:
            fast_mult = self._ema_multiplier(self.fast)
            slow_mult = self._ema_multiplier(self.slow)
            self._fast_ema = price * fast_mult + self._fast_ema * (1 - fast_mult)
            self._slow_ema = price * slow_mult + self._slow_ema * (1 - slow_mult)

        macd_line = self._fast_ema - self._slow_ema
        self._macd_values.append(macd_line)

        if len(self._macd_values) == 1:
            self._signal_ema = macd_line
        else:
            signal_mult = self._ema_multiplier(self.signal)
            self._signal_ema = macd_line * signal_mult + self._signal_ema * (1 - signal_mult)

        histogram = macd_line - self._signal_ema

        # Keep only necessary data
        if len(self._prices) > self.slow + self.signal + 10:
            self._prices = self._prices[-self.slow - self.signal :]
            self._macd_values = self._macd_values[-self.signal - 10 :]

        return macd_line, self._signal_ema, histogram

    def generate_signal(self, macd_line: float, signal_line: float, histogram: float) -> AlphaSignal:
        """
        Generate trading signal from MACD.

        Args:
            macd_line: MACD line value
            signal_line: Signal line value
            histogram: Histogram value

        Returns:
            AlphaSignal
        """
        # Detect crossovers
        if len(self._macd_values) >= 2:
            prev_hist = self._macd_values[-2] - self._signal_ema

            if prev_hist < 0 and histogram > 0:
                signal = Signal.BUY
                reason = "MACD bullish crossover"
                strength = 0.8
            elif prev_hist > 0 and histogram < 0:
                signal = Signal.SELL
                reason = "MACD bearish crossover"
                strength = 0.8
            else:
                if histogram > 0:
                    signal = Signal.BUY if macd_line > 0 else Signal.NEUTRAL
                    reason = "MACD positive histogram"
                else:
                    signal = Signal.SELL if macd_line < 0 else Signal.NEUTRAL
                    reason = "MACD negative histogram"
                strength = min(abs(histogram) / 2, 1.0)
        else:
            signal = Signal.NEUTRAL
            reason = "MACD initializing"
            strength = 0.0

        return AlphaSignal(
            indicator="MACD",
            signal=signal,
            strength=strength,
            value=histogram,
            reason=reason,
        )


class CCIIndicator:
    """
    Commodity Channel Index indicator.

    CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
    """

    def __init__(self, period: int = 20, overbought: float = 100, oversold: float = -100):
        """
        Initialize CCI indicator.

        Args:
            period: CCI period
            overbought: Overbought threshold
            oversold: Oversold threshold
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self._prices: list[float] = []

    @property
    def is_ready(self) -> bool:
        """Check if indicator has enough data."""
        return len(self._prices) >= self.period

    def update(self, high: float, low: float, close: float) -> float:
        """
        Update CCI with new data.

        Args:
            high: High price
            low: Low price
            close: Close price

        Returns:
            Current CCI value
        """
        typical_price = (high + low + close) / 3
        self._prices.append(typical_price)

        if len(self._prices) < self.period:
            return 0.0

        if len(self._prices) > self.period:
            self._prices = self._prices[-self.period :]

        sma = sum(self._prices) / self.period
        mean_deviation = sum(abs(p - sma) for p in self._prices) / self.period

        if mean_deviation == 0:
            return 0.0

        return (typical_price - sma) / (0.015 * mean_deviation)

    def generate_signal(self, cci: float) -> AlphaSignal:
        """
        Generate trading signal from CCI.

        Args:
            cci: Current CCI value

        Returns:
            AlphaSignal
        """
        if cci >= self.overbought:
            signal = Signal.SELL if cci < 200 else Signal.STRONG_SELL
            reason = f"CCI {cci:.1f} - overbought"
            strength = min(cci / 200, 1.0)
        elif cci <= self.oversold:
            signal = Signal.BUY if cci > -200 else Signal.STRONG_BUY
            reason = f"CCI {cci:.1f} - oversold"
            strength = min(abs(cci) / 200, 1.0)
        else:
            signal = Signal.NEUTRAL
            reason = f"CCI {cci:.1f} - neutral"
            strength = 0.0

        return AlphaSignal(
            indicator="CCI",
            signal=signal,
            strength=strength,
            value=cci,
            threshold=self.overbought if cci > 0 else self.oversold,
            reason=reason,
        )


class BollingerBandsIndicator:
    """
    Bollinger Bands indicator.

    Middle Band = SMA
    Upper Band = SMA + (std_dev * multiplier)
    Lower Band = SMA - (std_dev * multiplier)
    BBW = (Upper - Lower) / Middle
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands indicator.

        Args:
            period: SMA period
            std_dev: Standard deviation multiplier
        """
        self.period = period
        self.std_dev = std_dev
        self._prices: list[float] = []

    @property
    def is_ready(self) -> bool:
        """Check if indicator has enough data."""
        return len(self._prices) >= self.period

    def update(self, price: float) -> tuple[float, float, float, float]:
        """
        Update Bollinger Bands with new price.

        Args:
            price: Close price

        Returns:
            Tuple of (middle, upper, lower, BBW)
        """
        self._prices.append(price)

        if len(self._prices) < self.period:
            return price, price, price, 0.0

        if len(self._prices) > self.period:
            self._prices = self._prices[-self.period :]

        middle = sum(self._prices) / self.period
        variance = sum((p - middle) ** 2 for p in self._prices) / self.period
        std = variance**0.5

        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        bbw = (upper - lower) / middle if middle > 0 else 0

        return middle, upper, lower, bbw

    def generate_signal(self, price: float, middle: float, upper: float, lower: float) -> AlphaSignal:
        """
        Generate trading signal from Bollinger Bands.

        Args:
            price: Current price
            middle: Middle band
            upper: Upper band
            lower: Lower band

        Returns:
            AlphaSignal
        """
        band_width = upper - lower

        if price >= upper:
            signal = Signal.SELL
            reason = "Price at/above upper Bollinger Band"
            strength = min((price - upper) / (band_width * 0.5) + 0.5, 1.0)
        elif price <= lower:
            signal = Signal.BUY
            reason = "Price at/below lower Bollinger Band"
            strength = min((lower - price) / (band_width * 0.5) + 0.5, 1.0)
        elif price > middle:
            signal = Signal.NEUTRAL
            reason = "Price above middle band"
            strength = (price - middle) / (upper - middle) * 0.3
        else:
            signal = Signal.NEUTRAL
            reason = "Price below middle band"
            strength = (middle - price) / (middle - lower) * 0.3

        return AlphaSignal(
            indicator="Bollinger Bands",
            signal=signal,
            strength=strength,
            value=price,
            threshold=upper if price > middle else lower,
            reason=reason,
        )


class OBVIndicator:
    """
    On-Balance Volume indicator.

    OBV = Previous OBV + (Volume if close > prev_close, -Volume if close < prev_close)
    """

    def __init__(self, signal_period: int = 20):
        """
        Initialize OBV indicator.

        Args:
            signal_period: Period for OBV signal line
        """
        self.signal_period = signal_period
        self._obv = 0.0
        self._prev_close: float | None = None
        self._obv_values: list[float] = []

    @property
    def is_ready(self) -> bool:
        """Check if indicator has enough data."""
        return len(self._obv_values) >= self.signal_period

    def update(self, close: float, volume: int) -> float:
        """
        Update OBV with new data.

        Args:
            close: Close price
            volume: Volume

        Returns:
            Current OBV value
        """
        if self._prev_close is not None:
            if close > self._prev_close:
                self._obv += volume
            elif close < self._prev_close:
                self._obv -= volume

        self._prev_close = close
        self._obv_values.append(self._obv)

        if len(self._obv_values) > self.signal_period + 10:
            self._obv_values = self._obv_values[-self.signal_period - 10 :]

        return self._obv

    def generate_signal(self, obv: float, price_trend: int) -> AlphaSignal:
        """
        Generate trading signal from OBV.

        Args:
            obv: Current OBV value
            price_trend: 1 for up, -1 for down, 0 for neutral

        Returns:
            AlphaSignal
        """
        if len(self._obv_values) < 2:
            return AlphaSignal(
                indicator="OBV",
                signal=Signal.NEUTRAL,
                strength=0.0,
                value=obv,
                reason="OBV initializing",
            )

        obv_trend = 1 if self._obv_values[-1] > self._obv_values[-2] else -1

        # Look for divergences
        if price_trend > 0 and obv_trend < 0:
            signal = Signal.SELL
            reason = "Bearish divergence - price up, OBV down"
            strength = 0.7
        elif price_trend < 0 and obv_trend > 0:
            signal = Signal.BUY
            reason = "Bullish divergence - price down, OBV up"
            strength = 0.7
        elif price_trend > 0 and obv_trend > 0:
            signal = Signal.BUY
            reason = "OBV confirms uptrend"
            strength = 0.5
        elif price_trend < 0 and obv_trend < 0:
            signal = Signal.SELL
            reason = "OBV confirms downtrend"
            strength = 0.5
        else:
            signal = Signal.NEUTRAL
            reason = "No clear OBV signal"
            strength = 0.0

        return AlphaSignal(
            indicator="OBV",
            signal=signal,
            strength=strength,
            value=obv,
            reason=reason,
        )


class IchimokuIndicator:
    """
    Ichimoku Cloud indicator.

    Tenkan-sen = (Highest High + Lowest Low) / 2 for Tenkan period
    Kijun-sen = (Highest High + Lowest Low) / 2 for Kijun period
    Senkou Span A = (Tenkan + Kijun) / 2
    Senkou Span B = (Highest High + Lowest Low) / 2 for Senkou B period
    """

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        """
        Initialize Ichimoku indicator.

        Args:
            tenkan: Tenkan-sen period
            kijun: Kijun-sen period
            senkou_b: Senkou Span B period
        """
        self.tenkan_period = tenkan
        self.kijun_period = kijun
        self.senkou_b_period = senkou_b
        self._highs: list[float] = []
        self._lows: list[float] = []

    @property
    def is_ready(self) -> bool:
        """Check if indicator has enough data."""
        return len(self._highs) >= self.senkou_b_period

    def update(self, high: float, low: float) -> dict[str, float]:
        """
        Update Ichimoku with new data.

        Args:
            high: High price
            low: Low price

        Returns:
            Dictionary with Ichimoku values
        """
        self._highs.append(high)
        self._lows.append(low)

        if len(self._highs) > self.senkou_b_period + 26:
            self._highs = self._highs[-self.senkou_b_period - 26 :]
            self._lows = self._lows[-self.senkou_b_period - 26 :]

        result = {}

        if len(self._highs) >= self.tenkan_period:
            tenkan_high = max(self._highs[-self.tenkan_period :])
            tenkan_low = min(self._lows[-self.tenkan_period :])
            result["tenkan"] = (tenkan_high + tenkan_low) / 2

        if len(self._highs) >= self.kijun_period:
            kijun_high = max(self._highs[-self.kijun_period :])
            kijun_low = min(self._lows[-self.kijun_period :])
            result["kijun"] = (kijun_high + kijun_low) / 2

        if "tenkan" in result and "kijun" in result:
            result["senkou_a"] = (result["tenkan"] + result["kijun"]) / 2

        if len(self._highs) >= self.senkou_b_period:
            senkou_b_high = max(self._highs[-self.senkou_b_period :])
            senkou_b_low = min(self._lows[-self.senkou_b_period :])
            result["senkou_b"] = (senkou_b_high + senkou_b_low) / 2

        return result

    def generate_signal(self, price: float, ichimoku_values: dict[str, float]) -> AlphaSignal:
        """
        Generate trading signal from Ichimoku.

        Args:
            price: Current price
            ichimoku_values: Dictionary with Ichimoku values

        Returns:
            AlphaSignal
        """
        if "senkou_a" not in ichimoku_values or "senkou_b" not in ichimoku_values:
            return AlphaSignal(
                indicator="Ichimoku",
                signal=Signal.NEUTRAL,
                strength=0.0,
                value=price,
                reason="Ichimoku initializing",
            )

        senkou_a = ichimoku_values["senkou_a"]
        senkou_b = ichimoku_values["senkou_b"]
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        if price > cloud_top:
            signal = Signal.BUY
            reason = "Price above Ichimoku cloud - bullish"
            strength = min((price - cloud_top) / cloud_top * 10, 1.0)
        elif price < cloud_bottom:
            signal = Signal.SELL
            reason = "Price below Ichimoku cloud - bearish"
            strength = min((cloud_bottom - price) / cloud_bottom * 10, 1.0)
        else:
            signal = Signal.NEUTRAL
            reason = "Price inside Ichimoku cloud - indecision"
            strength = 0.3

        return AlphaSignal(
            indicator="Ichimoku",
            signal=signal,
            strength=strength,
            value=price,
            reason=reason,
        )


class TechnicalAlphaModel:
    """
    Combines multiple technical indicators for alpha generation.

    Aggregates signals from VWAP, RSI, MACD, CCI, Bollinger Bands,
    OBV, and Ichimoku for comprehensive technical analysis.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize technical alpha model.

        Args:
            config: Configuration for individual indicators
        """
        config = config or {}

        # Initialize indicators
        self.vwap = VWAPIndicator(anchor=config.get("vwap", {}).get("anchor", "session"))
        self.rsi = RSIIndicator(
            period=config.get("rsi", {}).get("period", 14),
            overbought=config.get("rsi", {}).get("overbought", 70),
            oversold=config.get("rsi", {}).get("oversold", 30),
        )
        self.macd = MACDIndicator(
            fast=config.get("macd", {}).get("fast", 12),
            slow=config.get("macd", {}).get("slow", 26),
            signal=config.get("macd", {}).get("signal", 9),
        )
        self.cci = CCIIndicator(
            period=config.get("cci", {}).get("period", 20),
            overbought=config.get("cci", {}).get("overbought", 100),
            oversold=config.get("cci", {}).get("oversold", -100),
        )
        self.bollinger = BollingerBandsIndicator(
            period=config.get("bollinger", {}).get("period", 20),
            std_dev=config.get("bollinger", {}).get("std_dev", 2.0),
        )
        self.obv = OBVIndicator(signal_period=config.get("obv", {}).get("signal_period", 20))
        self.ichimoku = IchimokuIndicator(
            tenkan=config.get("ichimoku", {}).get("tenkan", 9),
            kijun=config.get("ichimoku", {}).get("kijun", 26),
            senkou_b=config.get("ichimoku", {}).get("senkou_b", 52),
        )

        self._last_close: float | None = None

    def update(
        self,
        high: float,
        low: float,
        close: float,
        volume: int,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Update all indicators with new bar data.

        Args:
            high: High price
            low: Low price
            close: Close price
            volume: Volume
            timestamp: Bar timestamp

        Returns:
            Dictionary with all indicator values
        """
        typical_price = (high + low + close) / 3

        values = {}
        values["vwap"] = self.vwap.update(typical_price, volume, timestamp)
        values["rsi"] = self.rsi.update(close)
        values["macd"], values["macd_signal"], values["macd_histogram"] = self.macd.update(close)
        values["cci"] = self.cci.update(high, low, close)
        values["bb_middle"], values["bb_upper"], values["bb_lower"], values["bbw"] = self.bollinger.update(close)
        values["obv"] = self.obv.update(close, volume)
        values["ichimoku"] = self.ichimoku.update(high, low)

        self._last_close = close

        return values

    def generate_signals(self, current_price: float) -> list[AlphaSignal]:
        """
        Generate signals from all indicators.

        Args:
            current_price: Current market price

        Returns:
            List of AlphaSignal from each indicator
        """
        signals = []

        # VWAP signal
        vwap_value = (
            self.vwap._cumulative_pv / self.vwap._cumulative_volume
            if self.vwap._cumulative_volume > 0
            else current_price
        )
        signals.append(self.vwap.generate_signal(current_price, vwap_value))

        # RSI signal
        if self.rsi.is_ready:
            rsi_value = 50.0
            if self.rsi._avg_loss > 0:
                rs = self.rsi._avg_gain / self.rsi._avg_loss
                rsi_value = 100 - (100 / (1 + rs))
            signals.append(self.rsi.generate_signal(rsi_value))

        # MACD signal
        if self.macd.is_ready:
            macd_line = self.macd._fast_ema - self.macd._slow_ema
            histogram = macd_line - self.macd._signal_ema
            signals.append(self.macd.generate_signal(macd_line, self.macd._signal_ema, histogram))

        # CCI signal
        if self.cci.is_ready:
            cci_value = 0.0
            if self.cci._prices:
                sma = sum(self.cci._prices) / len(self.cci._prices)
                mean_dev = sum(abs(p - sma) for p in self.cci._prices) / len(self.cci._prices)
                if mean_dev > 0:
                    cci_value = (self.cci._prices[-1] - sma) / (0.015 * mean_dev)
            signals.append(self.cci.generate_signal(cci_value))

        # Bollinger Bands signal
        if self.bollinger.is_ready:
            middle = sum(self.bollinger._prices) / len(self.bollinger._prices)
            variance = sum((p - middle) ** 2 for p in self.bollinger._prices) / len(self.bollinger._prices)
            std = variance**0.5
            upper = middle + (std * self.bollinger.std_dev)
            lower = middle - (std * self.bollinger.std_dev)
            signals.append(self.bollinger.generate_signal(current_price, middle, upper, lower))

        # OBV signal
        if self.obv.is_ready:
            price_trend = 0
            if self._last_close and len(self.obv._obv_values) >= 2:
                if current_price > self._last_close:
                    price_trend = 1
                elif current_price < self._last_close:
                    price_trend = -1
            signals.append(self.obv.generate_signal(self.obv._obv, price_trend))

        # Ichimoku signal
        if self.ichimoku.is_ready:
            ichimoku_values = {}
            if len(self.ichimoku._highs) >= self.ichimoku.tenkan_period:
                tenkan_high = max(self.ichimoku._highs[-self.ichimoku.tenkan_period :])
                tenkan_low = min(self.ichimoku._lows[-self.ichimoku.tenkan_period :])
                ichimoku_values["tenkan"] = (tenkan_high + tenkan_low) / 2
            if len(self.ichimoku._highs) >= self.ichimoku.kijun_period:
                kijun_high = max(self.ichimoku._highs[-self.ichimoku.kijun_period :])
                kijun_low = min(self.ichimoku._lows[-self.ichimoku.kijun_period :])
                ichimoku_values["kijun"] = (kijun_high + kijun_low) / 2
            if "tenkan" in ichimoku_values and "kijun" in ichimoku_values:
                ichimoku_values["senkou_a"] = (ichimoku_values["tenkan"] + ichimoku_values["kijun"]) / 2
            if len(self.ichimoku._highs) >= self.ichimoku.senkou_b_period:
                senkou_b_high = max(self.ichimoku._highs[-self.ichimoku.senkou_b_period :])
                senkou_b_low = min(self.ichimoku._lows[-self.ichimoku.senkou_b_period :])
                ichimoku_values["senkou_b"] = (senkou_b_high + senkou_b_low) / 2
            signals.append(self.ichimoku.generate_signal(current_price, ichimoku_values))

        return signals

    def get_composite_signal(self, signals: list[AlphaSignal]) -> AlphaSignal:
        """
        Combine individual signals into composite signal.

        Args:
            signals: List of individual indicator signals

        Returns:
            Composite AlphaSignal
        """
        if not signals:
            return AlphaSignal(
                indicator="Composite",
                signal=Signal.NEUTRAL,
                strength=0.0,
                value=0.0,
                reason="No signals available",
            )

        # Weight signals by strength
        weighted_score = 0.0
        total_strength = 0.0

        for sig in signals:
            signal_value = sig.signal.value * sig.strength
            weighted_score += signal_value
            total_strength += sig.strength

        if total_strength > 0:
            avg_score = weighted_score / total_strength
        else:
            avg_score = 0.0

        # Determine composite signal
        if avg_score > 1.0:
            signal = Signal.STRONG_BUY
        elif avg_score > 0.3:
            signal = Signal.BUY
        elif avg_score < -1.0:
            signal = Signal.STRONG_SELL
        elif avg_score < -0.3:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        # Count signal agreement
        buy_count = sum(1 for s in signals if s.signal in (Signal.BUY, Signal.STRONG_BUY))
        sell_count = sum(1 for s in signals if s.signal in (Signal.SELL, Signal.STRONG_SELL))

        return AlphaSignal(
            indicator="Composite",
            signal=signal,
            strength=min(abs(avg_score), 1.0),
            value=avg_score,
            reason=f"{buy_count} buy, {sell_count} sell out of {len(signals)} indicators",
        )


class QuantConnectTechnicalAlphaModel:
    """
    Technical Alpha Model using QuantConnect's built-in indicators.

    INTEGRATION: Use this class when running in QuantConnect environment.
    Use TechnicalAlphaModel for standalone/non-QC environments.

    QuantConnect's built-in indicators are optimized, pre-tested, and
    automatically updated on each data point.

    Example:
        def Initialize(self):
            self.alpha_model = QuantConnectTechnicalAlphaModel(self, "SPY")

        def OnData(self, slice):
            if self.IsWarmingUp:
                return

            signals = self.alpha_model.generate_signals(slice)
            composite = self.alpha_model.get_composite_signal(signals)
    """

    def __init__(
        self,
        algorithm,
        symbol: str,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize with QuantConnect algorithm and symbol.

        INTEGRATION: Call this from algorithm.Initialize()

        Args:
            algorithm: QCAlgorithm instance
            symbol: Symbol to track (e.g., "SPY")
            config: Configuration for indicator parameters
        """
        self.algorithm = algorithm
        self.symbol = symbol
        config = config or {}

        # Subscribe to data if not already subscribed
        if not algorithm.Securities.ContainsKey(symbol):
            algorithm.AddEquity(symbol)

        # Create QuantConnect indicators using algorithm methods
        # These are automatically updated on each data point

        # RSI
        rsi_config = config.get("rsi", {})
        rsi_period = rsi_config.get("period", 14)
        self.rsi_overbought = rsi_config.get("overbought", 70)
        self.rsi_oversold = rsi_config.get("oversold", 30)
        try:
            from AlgorithmImports import MovingAverageType, Resolution

            self.rsi = algorithm.RSI(symbol, rsi_period, MovingAverageType.Wilders, Resolution.Daily)
        except ImportError:
            self.rsi = None

        # MACD
        macd_config = config.get("macd", {})
        macd_fast = macd_config.get("fast", 12)
        macd_slow = macd_config.get("slow", 26)
        macd_signal = macd_config.get("signal", 9)
        try:
            from AlgorithmImports import MovingAverageType, Resolution

            self.macd = algorithm.MACD(
                symbol, macd_fast, macd_slow, macd_signal, MovingAverageType.Exponential, Resolution.Daily
            )
        except ImportError:
            self.macd = None

        # Bollinger Bands
        bb_config = config.get("bollinger", {})
        bb_period = bb_config.get("period", 20)
        bb_std = bb_config.get("std_dev", 2.0)
        try:
            from AlgorithmImports import MovingAverageType, Resolution

            self.bollinger = algorithm.BB(symbol, bb_period, bb_std, MovingAverageType.Simple, Resolution.Daily)
        except ImportError:
            self.bollinger = None

        # CCI
        cci_config = config.get("cci", {})
        cci_period = cci_config.get("period", 20)
        self.cci_overbought = cci_config.get("overbought", 100)
        self.cci_oversold = cci_config.get("oversold", -100)
        try:
            from AlgorithmImports import Resolution

            self.cci = algorithm.CCI(symbol, cci_period, Resolution.Daily)
        except ImportError:
            self.cci = None

        # VWAP
        try:
            from AlgorithmImports import Resolution

            self.vwap = algorithm.VWAP(symbol, Resolution.Daily)
        except ImportError:
            self.vwap = None

        # Store thresholds
        self.config = config

        algorithm.Debug(f"QuantConnect Technical Alpha Model initialized for {symbol}")

    def generate_signals(self, slice) -> list[AlphaSignal]:
        """
        Generate signals from QuantConnect indicators.

        INTEGRATION: Call this from algorithm.OnData()

        Args:
            slice: Slice object from OnData

        Returns:
            List of AlphaSignal from each indicator
        """
        signals = []

        # Check if we have data for this symbol
        if self.symbol not in slice.Bars:
            return signals

        bar = slice.Bars[self.symbol]
        current_price = bar.Close

        # RSI signal
        if self.rsi and self.rsi.IsReady:
            rsi_value = self.rsi.Current.Value

            if rsi_value >= self.rsi_overbought:
                signal = Signal.SELL if rsi_value < 80 else Signal.STRONG_SELL
                reason = f"RSI {rsi_value:.1f} - overbought"
                strength = (rsi_value - self.rsi_overbought) / (100 - self.rsi_overbought)
            elif rsi_value <= self.rsi_oversold:
                signal = Signal.BUY if rsi_value > 20 else Signal.STRONG_BUY
                reason = f"RSI {rsi_value:.1f} - oversold"
                strength = (self.rsi_oversold - rsi_value) / self.rsi_oversold
            else:
                signal = Signal.NEUTRAL
                reason = f"RSI {rsi_value:.1f} - neutral"
                strength = 0.0

            signals.append(
                AlphaSignal(
                    indicator="RSI",
                    signal=signal,
                    strength=min(strength, 1.0),
                    value=rsi_value,
                    threshold=self.rsi_overbought if rsi_value > 50 else self.rsi_oversold,
                    reason=reason,
                )
            )

        # MACD signal
        if self.macd and self.macd.IsReady:
            macd_value = self.macd.Current.Value
            signal_value = self.macd.Signal.Current.Value
            histogram = macd_value - signal_value

            # Detect crossovers (need previous histogram)
            if histogram > 0 and abs(histogram) < 0.5:
                signal = Signal.BUY
                reason = "MACD bullish crossover"
                strength = 0.8
            elif histogram < 0 and abs(histogram) < 0.5:
                signal = Signal.SELL
                reason = "MACD bearish crossover"
                strength = 0.8
            else:
                if histogram > 0:
                    signal = Signal.BUY if macd_value > 0 else Signal.NEUTRAL
                    reason = "MACD positive histogram"
                else:
                    signal = Signal.SELL if macd_value < 0 else Signal.NEUTRAL
                    reason = "MACD negative histogram"
                strength = min(abs(histogram) / 2, 1.0)

            signals.append(
                AlphaSignal(
                    indicator="MACD",
                    signal=signal,
                    strength=strength,
                    value=histogram,
                    reason=reason,
                )
            )

        # Bollinger Bands signal
        if self.bollinger and self.bollinger.IsReady:
            upper = self.bollinger.UpperBand.Current.Value
            middle = self.bollinger.MiddleBand.Current.Value
            lower = self.bollinger.LowerBand.Current.Value
            band_width = upper - lower

            if current_price >= upper:
                signal = Signal.SELL
                reason = "Price at/above upper Bollinger Band"
                strength = min((current_price - upper) / (band_width * 0.5) + 0.5, 1.0)
            elif current_price <= lower:
                signal = Signal.BUY
                reason = "Price at/below lower Bollinger Band"
                strength = min((lower - current_price) / (band_width * 0.5) + 0.5, 1.0)
            elif current_price > middle:
                signal = Signal.NEUTRAL
                reason = "Price above middle band"
                strength = (current_price - middle) / (upper - middle) * 0.3
            else:
                signal = Signal.NEUTRAL
                reason = "Price below middle band"
                strength = (middle - current_price) / (middle - lower) * 0.3

            signals.append(
                AlphaSignal(
                    indicator="Bollinger Bands",
                    signal=signal,
                    strength=strength,
                    value=current_price,
                    threshold=upper if current_price > middle else lower,
                    reason=reason,
                )
            )

        # CCI signal
        if self.cci and self.cci.IsReady:
            cci_value = self.cci.Current.Value

            if cci_value >= self.cci_overbought:
                signal = Signal.SELL if cci_value < 200 else Signal.STRONG_SELL
                reason = f"CCI {cci_value:.1f} - overbought"
                strength = min(cci_value / 200, 1.0)
            elif cci_value <= self.cci_oversold:
                signal = Signal.BUY if cci_value > -200 else Signal.STRONG_BUY
                reason = f"CCI {cci_value:.1f} - oversold"
                strength = min(abs(cci_value) / 200, 1.0)
            else:
                signal = Signal.NEUTRAL
                reason = f"CCI {cci_value:.1f} - neutral"
                strength = 0.0

            signals.append(
                AlphaSignal(
                    indicator="CCI",
                    signal=signal,
                    strength=strength,
                    value=cci_value,
                    threshold=self.cci_overbought if cci_value > 0 else self.cci_oversold,
                    reason=reason,
                )
            )

        # VWAP signal
        if self.vwap and self.vwap.IsReady:
            vwap_value = self.vwap.Current.Value
            deviation = (current_price - vwap_value) / vwap_value if vwap_value > 0 else 0

            if deviation > 0.02:
                signal = Signal.SELL
                reason = f"Price {deviation:.1%} above VWAP - overbought"
            elif deviation < -0.02:
                signal = Signal.BUY
                reason = f"Price {deviation:.1%} below VWAP - oversold"
            else:
                signal = Signal.NEUTRAL
                reason = "Price near VWAP"

            signals.append(
                AlphaSignal(
                    indicator="VWAP",
                    signal=signal,
                    strength=min(abs(deviation) * 10, 1.0),
                    value=vwap_value,
                    reason=reason,
                )
            )

        return signals

    def get_composite_signal(self, signals: list[AlphaSignal]) -> AlphaSignal:
        """
        Combine individual signals into composite signal.

        Same logic as TechnicalAlphaModel.get_composite_signal().

        Args:
            signals: List of individual indicator signals

        Returns:
            Composite AlphaSignal
        """
        if not signals:
            return AlphaSignal(
                indicator="Composite",
                signal=Signal.NEUTRAL,
                strength=0.0,
                value=0.0,
                reason="No signals available",
            )

        # Weight signals by strength
        weighted_score = 0.0
        total_strength = 0.0

        for sig in signals:
            signal_value = sig.signal.value * sig.strength
            weighted_score += signal_value
            total_strength += sig.strength

        if total_strength > 0:
            avg_score = weighted_score / total_strength
        else:
            avg_score = 0.0

        # Determine composite signal
        if avg_score > 1.0:
            signal = Signal.STRONG_BUY
        elif avg_score > 0.3:
            signal = Signal.BUY
        elif avg_score < -1.0:
            signal = Signal.STRONG_SELL
        elif avg_score < -0.3:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        # Count signal agreement
        buy_count = sum(1 for s in signals if s.signal in (Signal.BUY, Signal.STRONG_BUY))
        sell_count = sum(1 for s in signals if s.signal in (Signal.SELL, Signal.STRONG_SELL))

        return AlphaSignal(
            indicator="Composite",
            signal=signal,
            strength=min(abs(avg_score), 1.0),
            value=avg_score,
            reason=f"{buy_count} buy, {sell_count} sell out of {len(signals)} indicators",
        )


__all__ = [
    "AlphaSignal",
    "BollingerBandsIndicator",
    "CCIIndicator",
    "IchimokuIndicator",
    "MACDIndicator",
    "OBVIndicator",
    "QuantConnectTechnicalAlphaModel",
    "RSIIndicator",
    "Signal",
    "TechnicalAlphaModel",
    "VWAPIndicator",
]
