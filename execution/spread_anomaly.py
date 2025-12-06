"""
Spread Anomaly Detection Module

Detects abnormal bid-ask spread behavior including:
- Quote stuffing patterns
- Market maker spread manipulation
- Sudden spread widening alerts
- Normal vs abnormal spread classification
- Spread history analysis for LLM/automated decisions

Based on research from FINRA, SEC, and market microstructure studies.
"""

import logging
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class SpreadAnomalyType(Enum):
    """Types of spread anomalies."""

    NORMAL = "normal"
    WIDE_SPREAD = "wide_spread"  # Spread wider than normal
    QUOTE_STUFFING = "quote_stuffing"  # Rapid quote changes
    SUDDEN_WIDENING = "sudden_widening"  # Rapid spread expansion
    LIQUIDITY_GAP = "liquidity_gap"  # Low depth on one side
    STALE_QUOTE = "stale_quote"  # Quote hasn't updated
    CROSSED_MARKET = "crossed_market"  # Bid > Ask (rare)


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class QuoteUpdate:
    """Single quote update for tracking changes."""

    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    sequence: int = 0  # For tracking quote frequency

    @property
    def spread(self) -> float:
        """Absolute spread."""
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        """Spread as percentage."""
        if self.mid == 0:
            return 0
        return self.spread / self.mid


@dataclass
class SpreadAnomaly:
    """Detected spread anomaly."""

    timestamp: datetime
    anomaly_type: SpreadAnomalyType
    severity: AnomalySeverity
    current_spread_bps: float
    normal_spread_bps: float
    deviation_factor: float  # How many times above normal
    description: str
    should_avoid_trading: bool
    fill_probability_impact: float  # Estimated impact on fill rate (-1 to 0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "type": self.anomaly_type.value,
            "severity": self.severity.value,
            "current_spread_bps": self.current_spread_bps,
            "normal_spread_bps": self.normal_spread_bps,
            "deviation_factor": self.deviation_factor,
            "description": self.description,
            "avoid_trading": self.should_avoid_trading,
            "fill_probability_impact": self.fill_probability_impact,
        }


@dataclass
class SpreadBaseline:
    """Baseline spread statistics for a symbol."""

    symbol: str
    avg_spread_bps: float
    median_spread_bps: float
    std_spread_bps: float
    p95_spread_bps: float  # 95th percentile
    p99_spread_bps: float  # 99th percentile
    avg_quote_frequency: float  # Quotes per second
    samples: int
    last_updated: datetime = field(default_factory=datetime.now)

    def is_spread_abnormal(self, current_spread_bps: float) -> tuple[bool, float]:
        """
        Check if current spread is abnormal.

        Returns (is_abnormal, deviation_factor)
        """
        if self.avg_spread_bps == 0:
            return False, 1.0

        deviation = current_spread_bps / self.avg_spread_bps

        # Consider abnormal if > 2 standard deviations or > 2x average
        threshold = max(
            self.avg_spread_bps + 2 * self.std_spread_bps,
            self.avg_spread_bps * 2,
        )

        is_abnormal = current_spread_bps > threshold
        return is_abnormal, deviation

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "avg_spread_bps": self.avg_spread_bps,
            "median_spread_bps": self.median_spread_bps,
            "std_spread_bps": self.std_spread_bps,
            "p95_spread_bps": self.p95_spread_bps,
            "p99_spread_bps": self.p99_spread_bps,
            "avg_quote_frequency": self.avg_quote_frequency,
            "samples": self.samples,
        }


class SpreadAnomalyDetector:
    """
    Detects abnormal spread behavior that may impact fill rates.

    Monitors for:
    - Quote stuffing (excessive quote updates)
    - Sudden spread widening
    - Market maker manipulation patterns
    - Liquidity gaps
    """

    def __init__(
        self,
        baseline_window: int = 1000,
        quote_stuffing_threshold: int = 50,  # Quotes per second
        sudden_widening_factor: float = 3.0,
        alert_callback: Callable[[SpreadAnomaly], None] | None = None,
    ):
        """
        Initialize detector.

        Args:
            baseline_window: Number of quotes for baseline calculation
            quote_stuffing_threshold: Quotes/second to trigger stuffing alert
            sudden_widening_factor: Spread increase factor for sudden widening
            alert_callback: Callback when anomaly detected
        """
        self.baseline_window = baseline_window
        self.quote_stuffing_threshold = quote_stuffing_threshold
        self.sudden_widening_factor = sudden_widening_factor
        self.alert_callback = alert_callback

        # Quote history per symbol
        self.quote_history: dict[str, deque] = {}
        self.baselines: dict[str, SpreadBaseline] = {}
        self.anomaly_history: dict[str, list[SpreadAnomaly]] = {}

        # Quote frequency tracking (for stuffing detection)
        self.quote_counts: dict[str, list[tuple[datetime, int]]] = {}

    def update(
        self,
        symbol: str,
        bid: float,
        ask: float,
        bid_size: int = 0,
        ask_size: int = 0,
        timestamp: datetime | None = None,
    ) -> SpreadAnomaly | None:
        """
        Update with new quote and check for anomalies.

        Args:
            symbol: Symbol
            bid: Bid price
            ask: Ask price
            bid_size: Bid size
            ask_size: Ask size
            timestamp: Quote timestamp

        Returns:
            SpreadAnomaly if detected, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Initialize if needed
        if symbol not in self.quote_history:
            self.quote_history[symbol] = deque(maxlen=self.baseline_window)
            self.quote_counts[symbol] = []
            self.anomaly_history[symbol] = []

        # Create quote update
        seq = len(self.quote_history[symbol])
        quote = QuoteUpdate(
            timestamp=timestamp,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            sequence=seq,
        )

        self.quote_history[symbol].append(quote)

        # Track quote frequency
        self._update_quote_frequency(symbol, timestamp)

        # Update baseline periodically
        if len(self.quote_history[symbol]) >= 100 and seq % 50 == 0:
            self._update_baseline(symbol)

        # Check for anomalies
        anomaly = self._detect_anomaly(symbol, quote)

        if anomaly:
            self.anomaly_history[symbol].append(anomaly)
            # Keep only recent anomalies
            if len(self.anomaly_history[symbol]) > 100:
                self.anomaly_history[symbol] = self.anomaly_history[symbol][-100:]

            if self.alert_callback:
                self.alert_callback(anomaly)

        return anomaly

    def _update_quote_frequency(self, symbol: str, timestamp: datetime) -> None:
        """Track quote frequency for stuffing detection."""
        self.quote_counts[symbol].append((timestamp, 1))

        # Keep only last 10 seconds
        cutoff = timestamp - timedelta(seconds=10)
        self.quote_counts[symbol] = [(t, c) for t, c in self.quote_counts[symbol] if t > cutoff]

    def _get_quote_frequency(self, symbol: str) -> float:
        """Get quotes per second over last 10 seconds."""
        if symbol not in self.quote_counts:
            return 0

        counts = self.quote_counts[symbol]
        if len(counts) < 2:
            return 0

        time_span = (counts[-1][0] - counts[0][0]).total_seconds()
        if time_span <= 0:
            return 0

        return len(counts) / time_span

    def _update_baseline(self, symbol: str) -> None:
        """Update baseline statistics from quote history."""
        quotes = list(self.quote_history[symbol])
        if len(quotes) < 50:
            return

        spreads_bps = [q.spread_pct * 10000 for q in quotes]

        # Sort for percentile calculations
        sorted_spreads = sorted(spreads_bps)

        avg_spread = sum(spreads_bps) / len(spreads_bps)
        median_idx = len(sorted_spreads) // 2
        median_spread = sorted_spreads[median_idx]

        # Standard deviation
        variance = sum((s - avg_spread) ** 2 for s in spreads_bps) / len(spreads_bps)
        std_spread = math.sqrt(variance)

        # Percentiles
        p95_idx = int(len(sorted_spreads) * 0.95)
        p99_idx = int(len(sorted_spreads) * 0.99)
        p95_spread = sorted_spreads[p95_idx] if p95_idx < len(sorted_spreads) else sorted_spreads[-1]
        p99_spread = sorted_spreads[p99_idx] if p99_idx < len(sorted_spreads) else sorted_spreads[-1]

        # Quote frequency
        avg_freq = self._get_quote_frequency(symbol)

        self.baselines[symbol] = SpreadBaseline(
            symbol=symbol,
            avg_spread_bps=avg_spread,
            median_spread_bps=median_spread,
            std_spread_bps=std_spread,
            p95_spread_bps=p95_spread,
            p99_spread_bps=p99_spread,
            avg_quote_frequency=avg_freq,
            samples=len(quotes),
        )

    def _detect_anomaly(
        self,
        symbol: str,
        quote: QuoteUpdate,
    ) -> SpreadAnomaly | None:
        """Detect anomalies in current quote."""
        anomalies = []

        current_spread_bps = quote.spread_pct * 10000

        # Check for crossed market (critical)
        if quote.bid > quote.ask:
            return SpreadAnomaly(
                timestamp=quote.timestamp,
                anomaly_type=SpreadAnomalyType.CROSSED_MARKET,
                severity=AnomalySeverity.CRITICAL,
                current_spread_bps=current_spread_bps,
                normal_spread_bps=0,
                deviation_factor=float("inf"),
                description=f"Crossed market: bid {quote.bid} > ask {quote.ask}",
                should_avoid_trading=True,
                fill_probability_impact=-1.0,
            )

        # Check against baseline
        if symbol in self.baselines:
            baseline = self.baselines[symbol]
            is_abnormal, deviation = baseline.is_spread_abnormal(current_spread_bps)

            if is_abnormal:
                # Determine severity
                if deviation > 5:
                    severity = AnomalySeverity.CRITICAL
                    should_avoid = True
                    fill_impact = -0.8
                elif deviation > 3:
                    severity = AnomalySeverity.WARNING
                    should_avoid = True
                    fill_impact = -0.5
                else:
                    severity = AnomalySeverity.INFO
                    should_avoid = False
                    fill_impact = -0.2

                anomalies.append(
                    SpreadAnomaly(
                        timestamp=quote.timestamp,
                        anomaly_type=SpreadAnomalyType.WIDE_SPREAD,
                        severity=severity,
                        current_spread_bps=current_spread_bps,
                        normal_spread_bps=baseline.avg_spread_bps,
                        deviation_factor=deviation,
                        description=f"Spread {deviation:.1f}x wider than normal",
                        should_avoid_trading=should_avoid,
                        fill_probability_impact=fill_impact,
                    )
                )

        # Check for quote stuffing
        quote_freq = self._get_quote_frequency(symbol)
        if quote_freq > self.quote_stuffing_threshold:
            anomalies.append(
                SpreadAnomaly(
                    timestamp=quote.timestamp,
                    anomaly_type=SpreadAnomalyType.QUOTE_STUFFING,
                    severity=AnomalySeverity.WARNING,
                    current_spread_bps=current_spread_bps,
                    normal_spread_bps=self.baselines.get(
                        symbol,
                        SpreadBaseline(
                            symbol=symbol,
                            avg_spread_bps=0,
                            median_spread_bps=0,
                            std_spread_bps=0,
                            p95_spread_bps=0,
                            p99_spread_bps=0,
                            avg_quote_frequency=0,
                            samples=0,
                        ),
                    ).avg_spread_bps,
                    deviation_factor=quote_freq / self.quote_stuffing_threshold,
                    description=f"Quote stuffing detected: {quote_freq:.0f} quotes/sec",
                    should_avoid_trading=True,
                    fill_probability_impact=-0.6,
                )
            )

        # Check for sudden widening
        if len(self.quote_history[symbol]) >= 5:
            recent_quotes = list(self.quote_history[symbol])[-5:]
            recent_spreads = [q.spread_pct * 10000 for q in recent_quotes[:-1]]
            avg_recent = sum(recent_spreads) / len(recent_spreads) if recent_spreads else 0

            if avg_recent > 0 and current_spread_bps > avg_recent * self.sudden_widening_factor:
                widening_factor = current_spread_bps / avg_recent
                anomalies.append(
                    SpreadAnomaly(
                        timestamp=quote.timestamp,
                        anomaly_type=SpreadAnomalyType.SUDDEN_WIDENING,
                        severity=AnomalySeverity.WARNING,
                        current_spread_bps=current_spread_bps,
                        normal_spread_bps=avg_recent,
                        deviation_factor=widening_factor,
                        description=f"Sudden spread widening: {widening_factor:.1f}x in 5 quotes",
                        should_avoid_trading=True,
                        fill_probability_impact=-0.7,
                    )
                )

        # Check for liquidity gaps
        if quote.bid_size > 0 and quote.ask_size > 0:
            size_ratio = max(quote.bid_size, quote.ask_size) / min(quote.bid_size, quote.ask_size)
            if size_ratio > 10:  # One side has 10x more size
                thin_side = "ask" if quote.ask_size < quote.bid_size else "bid"
                anomalies.append(
                    SpreadAnomaly(
                        timestamp=quote.timestamp,
                        anomaly_type=SpreadAnomalyType.LIQUIDITY_GAP,
                        severity=AnomalySeverity.INFO,
                        current_spread_bps=current_spread_bps,
                        normal_spread_bps=self.baselines.get(
                            symbol,
                            SpreadBaseline(
                                symbol=symbol,
                                avg_spread_bps=0,
                                median_spread_bps=0,
                                std_spread_bps=0,
                                p95_spread_bps=0,
                                p99_spread_bps=0,
                                avg_quote_frequency=0,
                                samples=0,
                            ),
                        ).avg_spread_bps,
                        deviation_factor=size_ratio,
                        description=f"Liquidity gap: {thin_side} side thin ({size_ratio:.0f}x imbalance)",
                        should_avoid_trading=False,
                        fill_probability_impact=-0.3,
                    )
                )

        # Return most severe anomaly
        if anomalies:
            severity_order = {
                AnomalySeverity.CRITICAL: 3,
                AnomalySeverity.WARNING: 2,
                AnomalySeverity.INFO: 1,
            }
            anomalies.sort(key=lambda a: severity_order[a.severity], reverse=True)
            return anomalies[0]

        return None

    def is_safe_to_trade(
        self,
        symbol: str,
        lookback_seconds: int = 30,
    ) -> tuple[bool, str, float]:
        """
        Check if it's safe to trade based on recent spread behavior.

        Args:
            symbol: Symbol to check
            lookback_seconds: Seconds of history to consider

        Returns:
            (is_safe, reason, confidence)
        """
        if symbol not in self.anomaly_history:
            return True, "No anomaly history", 0.5

        cutoff = datetime.now() - timedelta(seconds=lookback_seconds)
        recent_anomalies = [a for a in self.anomaly_history[symbol] if a.timestamp > cutoff]

        if not recent_anomalies:
            # Check current spread against baseline
            if symbol in self.baselines and symbol in self.quote_history:
                quotes = list(self.quote_history[symbol])
                if quotes:
                    current = quotes[-1]
                    is_abnormal, deviation = self.baselines[symbol].is_spread_abnormal(current.spread_pct * 10000)
                    if is_abnormal:
                        return False, f"Spread {deviation:.1f}x above normal", 0.7
            return True, "No recent anomalies", 0.8

        # Check for critical anomalies
        critical = [a for a in recent_anomalies if a.severity == AnomalySeverity.CRITICAL]
        if critical:
            return False, critical[0].description, 0.95

        # Check for multiple warnings
        warnings = [a for a in recent_anomalies if a.severity == AnomalySeverity.WARNING]
        if len(warnings) >= 2:
            return False, f"Multiple warnings: {warnings[0].anomaly_type.value}", 0.8

        # Single warning - cautious but allowed
        if warnings:
            return True, f"Caution: {warnings[0].description}", 0.6

        return True, "Minor anomalies only", 0.7

    def get_spread_quality_score(self, symbol: str) -> float:
        """
        Get overall spread quality score (0-100).

        Higher = better conditions for trading.
        """
        if symbol not in self.quote_history:
            return 50.0

        quotes = list(self.quote_history[symbol])
        if not quotes:
            return 50.0

        current = quotes[-1]
        score = 100.0

        # Penalize for wide spreads
        if symbol in self.baselines:
            baseline = self.baselines[symbol]
            current_bps = current.spread_pct * 10000
            if baseline.avg_spread_bps > 0:
                spread_ratio = current_bps / baseline.avg_spread_bps
                if spread_ratio > 1:
                    score -= min(40, (spread_ratio - 1) * 20)

        # Penalize for recent anomalies
        cutoff = datetime.now() - timedelta(seconds=60)
        recent_anomalies = [a for a in self.anomaly_history.get(symbol, []) if a.timestamp > cutoff]

        for anomaly in recent_anomalies:
            if anomaly.severity == AnomalySeverity.CRITICAL:
                score -= 30
            elif anomaly.severity == AnomalySeverity.WARNING:
                score -= 15
            else:
                score -= 5

        # Penalize for high quote frequency (potential stuffing)
        quote_freq = self._get_quote_frequency(symbol)
        if quote_freq > self.quote_stuffing_threshold * 0.5:
            score -= 10

        return max(0, min(100, score))

    def get_baseline(self, symbol: str) -> SpreadBaseline | None:
        """Get current baseline for symbol."""
        return self.baselines.get(symbol)

    def get_recent_anomalies(
        self,
        symbol: str,
        limit: int = 10,
    ) -> list[SpreadAnomaly]:
        """Get recent anomalies for symbol."""
        if symbol not in self.anomaly_history:
            return []
        return self.anomaly_history[symbol][-limit:]

    def get_summary(self, symbol: str) -> dict[str, Any]:
        """Get comprehensive spread analysis summary."""
        baseline = self.baselines.get(symbol)
        recent_anomalies = self.get_recent_anomalies(symbol, 10)
        is_safe, reason, confidence = self.is_safe_to_trade(symbol)

        current_quote = None
        if symbol in self.quote_history:
            quotes = list(self.quote_history[symbol])
            if quotes:
                current_quote = quotes[-1]

        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_spread_bps": current_quote.spread_pct * 10000 if current_quote else 0,
            "baseline": baseline.to_dict() if baseline else None,
            "spread_quality_score": self.get_spread_quality_score(symbol),
            "is_safe_to_trade": is_safe,
            "safety_reason": reason,
            "safety_confidence": confidence,
            "quote_frequency": self._get_quote_frequency(symbol),
            "recent_anomalies": [a.to_dict() for a in recent_anomalies],
        }

    def get_llm_summary(self, symbol: str) -> str:
        """Generate LLM-ready summary of spread conditions."""
        summary = self.get_summary(symbol)

        text = f"""
SPREAD ANALYSIS FOR {symbol}
============================
Current Spread: {summary['current_spread_bps']:.1f} bps
Spread Quality Score: {summary['spread_quality_score']:.0f}/100
Safe to Trade: {'YES' if summary['is_safe_to_trade'] else 'NO'} ({summary['safety_reason']})
Quote Frequency: {summary['quote_frequency']:.1f} quotes/sec
"""

        if summary["baseline"]:
            b = summary["baseline"]
            text += f"""
BASELINE STATISTICS
-------------------
Average Spread: {b['avg_spread_bps']:.1f} bps
Median Spread: {b['median_spread_bps']:.1f} bps
95th Percentile: {b['p95_spread_bps']:.1f} bps
99th Percentile: {b['p99_spread_bps']:.1f} bps
"""

        if summary["recent_anomalies"]:
            text += "\nRECENT ANOMALIES\n----------------\n"
            for a in summary["recent_anomalies"][-5:]:
                text += f"- [{a['severity'].upper()}] {a['type']}: {a['description']}\n"

        return text


def create_spread_anomaly_detector(
    baseline_window: int = 1000,
    quote_stuffing_threshold: int = 50,
    sudden_widening_factor: float = 3.0,
    alert_callback: Callable[[SpreadAnomaly], None] | None = None,
) -> SpreadAnomalyDetector:
    """Create spread anomaly detector instance."""
    return SpreadAnomalyDetector(
        baseline_window=baseline_window,
        quote_stuffing_threshold=quote_stuffing_threshold,
        sudden_widening_factor=sudden_widening_factor,
        alert_callback=alert_callback,
    )


__all__ = [
    "AnomalySeverity",
    "QuoteUpdate",
    "SpreadAnomaly",
    "SpreadAnomalyDetector",
    "SpreadAnomalyType",
    "SpreadBaseline",
    "create_spread_anomaly_detector",
]
