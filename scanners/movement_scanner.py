"""
Movement Scanner Module

Scans for significant price movements with news corroboration.
Identifies potential trading opportunities from unusual price action.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from config import MovementScannerConfig
from llm import NewsAnalyzer, NewsItem, SentimentResult


logger = logging.getLogger(__name__)


class MovementDirection(Enum):
    """Direction of price movement."""

    UP = "up"
    DOWN = "down"


@dataclass
class PriceData:
    """Price data for a symbol."""

    symbol: str
    current_price: float
    open_price: float
    previous_close: float
    high: float
    low: float
    volume: int
    average_volume: int
    timestamp: datetime

    @property
    def change_from_open(self) -> float:
        """Percentage change from open."""
        if self.open_price > 0:
            return (self.current_price - self.open_price) / self.open_price
        return 0.0

    @property
    def change_from_close(self) -> float:
        """Percentage change from previous close."""
        if self.previous_close > 0:
            return (self.current_price - self.previous_close) / self.previous_close
        return 0.0

    @property
    def volume_ratio(self) -> float:
        """Current volume vs average volume."""
        if self.average_volume > 0:
            return self.volume / self.average_volume
        return 1.0

    @property
    def intraday_range(self) -> float:
        """Intraday price range as percentage."""
        if self.low > 0:
            return (self.high - self.low) / self.low
        return 0.0


@dataclass
class MovementAlert:
    """Alert generated from significant price movement."""

    symbol: str
    direction: MovementDirection
    movement_pct: float
    volume_ratio: float
    price_data: PriceData
    news_corroboration: bool
    related_news: list[NewsItem] = field(default_factory=list)
    sentiment: SentimentResult | None = None
    suggested_options: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "movement_pct": self.movement_pct,
            "volume_ratio": self.volume_ratio,
            "current_price": self.price_data.current_price,
            "news_corroboration": self.news_corroboration,
            "related_news_count": len(self.related_news),
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "suggested_options": self.suggested_options,
            "timestamp": self.timestamp.isoformat(),
        }


class MovementScanner:
    """
    Scans for significant price movements with optional news corroboration.

    Identifies stocks with unusual price action and correlates with
    recent news to validate the move.
    """

    def __init__(
        self,
        config: MovementScannerConfig,
        news_analyzer: NewsAnalyzer | None = None,
        alert_callback: Callable[[MovementAlert], None] | None = None,
    ):
        """
        Initialize movement scanner.

        Args:
            config: Scanner configuration
            news_analyzer: News analyzer for corroboration
            alert_callback: Callback for movement alerts
        """
        self.config = config
        self.news_analyzer = news_analyzer
        self.alert_callback = alert_callback

        self._price_history: dict[str, list[PriceData]] = {}
        self._alerted_moves: dict[str, datetime] = {}

    def _should_alert(self, symbol: str) -> bool:
        """Check if we should alert for this symbol (avoid spam)."""
        last_alert = self._alerted_moves.get(symbol)
        if last_alert is None:
            return True

        # Don't alert again within 15 minutes
        if datetime.now() - last_alert < timedelta(minutes=15):
            return False

        return True

    def _is_significant_movement(self, price_data: PriceData) -> tuple:
        """
        Check if price movement is significant.

        Returns:
            Tuple of (is_significant, direction, movement_pct)
        """
        movement_pct = abs(price_data.change_from_close)

        if movement_pct < self.config.min_movement_pct:
            return False, None, 0.0

        if movement_pct > self.config.max_movement_pct:
            # Very large moves might be errors or corporate actions
            return False, None, 0.0

        direction = MovementDirection.UP if price_data.change_from_close > 0 else MovementDirection.DOWN

        return True, direction, movement_pct

    def _has_volume_surge(self, price_data: PriceData) -> bool:
        """Check if there's unusual volume."""
        return price_data.volume_ratio >= self.config.volume_surge_threshold

    def _get_related_news(self, symbol: str, hours_back: int = 24) -> list[NewsItem]:
        """Get recent news for symbol."""
        if self.news_analyzer is None:
            return []

        return self.news_analyzer.get_recent_news(symbol=symbol, hours=hours_back)

    def _corroborate_with_news(self, symbol: str, direction: MovementDirection) -> tuple:
        """
        Check if movement is corroborated by news.

        Returns:
            Tuple of (is_corroborated, related_news, sentiment)
        """
        if not self.config.require_news_corroboration:
            return True, [], None

        if self.news_analyzer is None:
            return True, [], None  # No news analyzer, assume corroborated

        related_news = self._get_related_news(symbol)

        if not related_news:
            return False, [], None

        # Analyze sentiment of related news
        from llm import create_ensemble

        ensemble = create_ensemble()
        all_text = " ".join([n.title for n in related_news[:5]])
        sentiment_result = ensemble.analyze_sentiment(all_text)

        # Check if sentiment aligns with movement
        if direction == MovementDirection.UP:
            is_corroborated = sentiment_result.sentiment.score > 0
        else:
            is_corroborated = sentiment_result.sentiment.score < 0

        return is_corroborated, related_news, sentiment_result.sentiment

    def _suggest_options(self, symbol: str, direction: MovementDirection, movement_pct: float) -> list[dict[str, Any]]:
        """
        Suggest options based on movement.

        Returns simple suggestions - actual option selection
        should use OptionsScanner for detailed analysis.
        """
        suggestions = []

        if direction == MovementDirection.UP:
            suggestions.append(
                {
                    "type": "call",
                    "strategy": "long_call",
                    "reasoning": f"{symbol} showing {movement_pct:.1%} upward movement",
                    "delta_target": 0.30,
                }
            )
            suggestions.append(
                {
                    "type": "put",
                    "strategy": "sell_put",
                    "reasoning": f"Bullish momentum, collect premium on {symbol}",
                    "delta_target": -0.25,
                }
            )
        else:
            suggestions.append(
                {
                    "type": "put",
                    "strategy": "long_put",
                    "reasoning": f"{symbol} showing {movement_pct:.1%} downward movement",
                    "delta_target": -0.30,
                }
            )
            suggestions.append(
                {
                    "type": "call",
                    "strategy": "sell_call",
                    "reasoning": f"Bearish momentum, collect premium on {symbol}",
                    "delta_target": 0.25,
                }
            )

        return suggestions

    def scan(self, price_data: PriceData) -> MovementAlert | None:
        """
        Scan single symbol for significant movement.

        Args:
            price_data: Current price data for symbol

        Returns:
            MovementAlert if significant movement found, None otherwise
        """
        if not self.config.enabled:
            return None

        symbol = price_data.symbol

        # Check if movement is significant
        is_significant, direction, movement_pct = self._is_significant_movement(price_data)
        if not is_significant:
            return None

        # Check volume
        has_volume = self._has_volume_surge(price_data)
        if not has_volume:
            return None

        # Check if we should alert (avoid spam)
        if not self._should_alert(symbol):
            return None

        # Corroborate with news
        is_corroborated, related_news, sentiment = self._corroborate_with_news(symbol, direction)

        if self.config.require_news_corroboration and not is_corroborated:
            return None

        # Generate option suggestions
        suggested_options = self._suggest_options(symbol, direction, movement_pct)

        # Create alert
        alert = MovementAlert(
            symbol=symbol,
            direction=direction,
            movement_pct=movement_pct,
            volume_ratio=price_data.volume_ratio,
            price_data=price_data,
            news_corroboration=is_corroborated,
            related_news=related_news,
            sentiment=sentiment,
            suggested_options=suggested_options,
        )

        # Record alert
        self._alerted_moves[symbol] = datetime.now()

        # Trigger callback
        if self.alert_callback:
            self.alert_callback(alert)

        return alert

    def scan_batch(self, price_data_list: list[PriceData]) -> list[MovementAlert]:
        """
        Scan multiple symbols for significant movements.

        Args:
            price_data_list: List of price data to scan

        Returns:
            List of movement alerts
        """
        alerts = []

        for price_data in price_data_list:
            alert = self.scan(price_data)
            if alert:
                alerts.append(alert)

        # Sort by movement magnitude
        alerts.sort(key=lambda x: abs(x.movement_pct), reverse=True)

        return alerts

    def update_price_history(self, price_data: PriceData) -> None:
        """Update price history for symbol."""
        symbol = price_data.symbol

        if symbol not in self._price_history:
            self._price_history[symbol] = []

        self._price_history[symbol].append(price_data)

        # Keep last 100 data points
        if len(self._price_history[symbol]) > 100:
            self._price_history[symbol] = self._price_history[symbol][-100:]

    def get_top_movers(self, n: int = 10, direction: MovementDirection | None = None) -> list[PriceData]:
        """
        Get top movers from recent price data.

        Args:
            n: Number of movers to return
            direction: Filter by direction (optional)

        Returns:
            List of top moving stocks
        """
        all_prices = []
        for prices in self._price_history.values():
            if prices:
                all_prices.append(prices[-1])

        if direction == MovementDirection.UP:
            all_prices = [p for p in all_prices if p.change_from_close > 0]
        elif direction == MovementDirection.DOWN:
            all_prices = [p for p in all_prices if p.change_from_close < 0]

        all_prices.sort(key=lambda x: abs(x.change_from_close), reverse=True)

        return all_prices[:n]

    def scan_from_qc_slice(
        self,
        algorithm,
        slice,
        symbols: list[str],
    ) -> list[MovementAlert]:
        """
        Scan for movements from QuantConnect OnData slice with data validation.

        INTEGRATION: Call this from algorithm.OnData()

        Example:
            def OnData(self, slice):
                if self.IsWarmingUp:
                    return

                watchlist = ["SPY", "QQQ", "AAPL", "TSLA"]
                alerts = self.movement_scanner.scan_from_qc_slice(self, slice, watchlist)
                for alert in alerts:
                    self.Debug(f"Movement: {alert.symbol} {alert.movement_pct:+.2%}")

        Args:
            algorithm: QCAlgorithm instance
            slice: Slice object from OnData
            symbols: List of symbols to scan

        Returns:
            List of movement alerts
        """
        price_data_list = []

        for symbol in symbols:
            # Data validation: check if we have bar data
            if symbol not in slice.Bars:
                continue

            bar = slice.Bars[symbol]

            # Validate data quality
            if bar.Close <= 0 or bar.Open <= 0:
                algorithm.Debug(f"Invalid price data for {symbol}: Close={bar.Close}, Open={bar.Open}")
                continue

            if bar.Volume <= 0:
                # Skip if no volume data
                continue

            # Get historical data for comparison
            if algorithm.Securities.ContainsKey(symbol):
                security = algorithm.Securities[symbol]

                # Get previous close using History API
                # NOTE: Security.Close is the CURRENT close, NOT previous day's close
                # For accurate previous close, use History
                from AlgorithmImports import Resolution

                history = algorithm.History([symbol], 2, Resolution.Daily)
                if not history.empty and len(history) >= 2:
                    prev_close = float(history["close"].iloc[-2])
                else:
                    # Fallback: use current price (not ideal for change calculation)
                    prev_close = security.Price

                # Get average volume (approximation - you may want to track this separately)
                avg_volume = bar.Volume  # Simplified - ideally track 20-day average

                price_data = PriceData(
                    symbol=symbol,
                    current_price=bar.Close,
                    open_price=bar.Open,
                    previous_close=prev_close,
                    high=bar.High,
                    low=bar.Low,
                    volume=int(bar.Volume),
                    average_volume=int(avg_volume),
                    timestamp=algorithm.Time,
                )
                # Note: change_from_close and volume_ratio are calculated automatically as @properties

                price_data_list.append(price_data)

        # Use existing batch scan logic
        return self.scan_batch(price_data_list)


def create_movement_scanner(
    config: MovementScannerConfig | None = None,
    news_analyzer: NewsAnalyzer | None = None,
    alert_callback: Callable[[MovementAlert], None] | None = None,
) -> MovementScanner:
    """
    Create movement scanner from configuration.

    Args:
        config: Scanner configuration
        news_analyzer: News analyzer for corroboration
        alert_callback: Optional callback for alerts

    Returns:
        Configured MovementScanner instance
    """
    if config is None:
        config = MovementScannerConfig()

    return MovementScanner(config, news_analyzer=news_analyzer, alert_callback=alert_callback)


__all__ = [
    "MovementAlert",
    "MovementDirection",
    "MovementScanner",
    "PriceData",
    "create_movement_scanner",
]
