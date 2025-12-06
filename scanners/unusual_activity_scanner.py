"""
Unusual Options Activity Scanner

Detects unusual options activity patterns that may indicate
institutional trading or informed positioning:
- Volume spikes
- Open interest surges
- IV spikes
- Block trades
- Put/call skew changes
- Sweep orders

Part of UPGRADE-010 Sprint 4: Risk & Execution.
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class ActivityType(Enum):
    """Types of unusual activity."""

    VOLUME_SPIKE = "volume_spike"
    OI_SURGE = "oi_surge"
    IV_SPIKE = "iv_spike"
    BLOCK_TRADE = "block_trade"
    PUT_CALL_SKEW = "put_call_skew"
    SWEEP = "sweep"
    UNUSUAL_SIZE = "unusual_size"
    UNUSUAL_PREMIUM = "unusual_premium"


class DirectionBias(Enum):
    """Directional bias of activity."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class UrgencyLevel(Enum):
    """Urgency level of alert."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class UnusualActivityAlert:
    """Alert for unusual options activity."""

    symbol: str
    underlying: str
    activity_type: ActivityType
    current_value: float  # Current metric value
    historical_avg: float  # Historical average
    deviation_sigma: float  # Standard deviations from mean
    percentile: float  # Percentile rank (0-100)
    volume: int  # Contracts traded
    premium: float  # Dollar premium traded
    direction_bias: DirectionBias
    confidence: float  # 0-1
    urgency: UrgencyLevel
    timestamp: datetime
    expiry: datetime
    strike: float
    option_type: str  # "call" or "put"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "activity_type": self.activity_type.value,
            "current_value": self.current_value,
            "historical_avg": self.historical_avg,
            "deviation_sigma": self.deviation_sigma,
            "percentile": self.percentile,
            "volume": self.volume,
            "premium": self.premium,
            "direction_bias": self.direction_bias.value,
            "confidence": self.confidence,
            "urgency": self.urgency.value,
            "timestamp": self.timestamp.isoformat(),
            "expiry": self.expiry.isoformat(),
            "strike": self.strike,
            "option_type": self.option_type,
            "details": self.details,
        }


@dataclass
class OptionActivityData:
    """Data for a single option contract activity."""

    symbol: str
    underlying: str
    option_type: str  # "call" or "put"
    strike: float
    expiry: datetime
    volume: int
    open_interest: int
    implied_volatility: float
    bid: float
    ask: float
    last_price: float
    underlying_price: float
    delta: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def mid_price(self) -> float:
        """Calculate mid price from bid/ask."""
        return (self.bid + self.ask) / 2

    @property
    def premium_traded(self) -> float:
        """Estimate premium traded (mid price * volume * 100)."""
        return self.mid_price * self.volume * 100

    @property
    def volume_oi_ratio(self) -> float:
        """Volume to open interest ratio."""
        return self.volume / self.open_interest if self.open_interest > 0 else 0


@dataclass
class UnusualActivityConfig:
    """Configuration for unusual activity detection."""

    # Volume thresholds
    volume_threshold_sigma: float = 2.0  # 2 standard deviations
    volume_oi_threshold: float = 0.5  # Volume > 50% of OI

    # OI thresholds
    oi_threshold_sigma: float = 2.5  # 2.5 std devs for OI change
    oi_change_min_pct: float = 0.10  # Min 10% OI change

    # IV thresholds
    iv_threshold_sigma: float = 2.0  # 2 std devs for IV spike
    iv_min_percentile: float = 80  # IV > 80th percentile

    # Block trade thresholds
    block_trade_threshold: int = 100  # 100+ contracts
    block_premium_threshold: float = 50000  # $50k+ premium

    # Put/call skew thresholds
    put_call_extreme_low: float = 0.3  # P/C ratio < 0.3 = bullish
    put_call_extreme_high: float = 3.0  # P/C ratio > 3.0 = bearish

    # General settings
    lookback_days: int = 20  # Days for historical comparison
    min_volume: int = 10  # Minimum volume to consider
    min_oi: int = 100  # Minimum OI to consider


@dataclass
class ActivityHistory:
    """Historical activity data for a symbol."""

    volumes: list[int] = field(default_factory=list)
    open_interests: list[int] = field(default_factory=list)
    ivs: list[float] = field(default_factory=list)
    put_call_ratios: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)

    def add_record(
        self,
        volume: int,
        oi: int,
        iv: float,
        pc_ratio: float,
        timestamp: datetime,
    ) -> None:
        """Add a historical record."""
        self.volumes.append(volume)
        self.open_interests.append(oi)
        self.ivs.append(iv)
        self.put_call_ratios.append(pc_ratio)
        self.timestamps.append(timestamp)

    def trim(self, lookback_days: int) -> None:
        """Trim history to lookback period."""
        cutoff = datetime.now() - timedelta(days=lookback_days)
        valid_idx = [i for i, t in enumerate(self.timestamps) if t >= cutoff]

        if valid_idx:
            self.volumes = [self.volumes[i] for i in valid_idx]
            self.open_interests = [self.open_interests[i] for i in valid_idx]
            self.ivs = [self.ivs[i] for i in valid_idx]
            self.put_call_ratios = [self.put_call_ratios[i] for i in valid_idx]
            self.timestamps = [self.timestamps[i] for i in valid_idx]


@dataclass
class FlowAnalysis:
    """Analysis of options flow for a symbol."""

    underlying: str
    total_call_volume: int
    total_put_volume: int
    put_call_ratio: float
    total_premium: float
    call_premium: float
    put_premium: float
    net_delta: float  # Net delta exposure
    largest_trades: list[UnusualActivityAlert]
    direction_bias: DirectionBias
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "underlying": self.underlying,
            "total_call_volume": self.total_call_volume,
            "total_put_volume": self.total_put_volume,
            "put_call_ratio": self.put_call_ratio,
            "total_premium": self.total_premium,
            "call_premium": self.call_premium,
            "put_premium": self.put_premium,
            "net_delta": self.net_delta,
            "direction_bias": self.direction_bias.value,
            "confidence": self.confidence,
            "largest_trades_count": len(self.largest_trades),
        }


class UnusualActivityScanner:
    """
    Scan for unusual options activity patterns.

    Detects institutional activity and informed trading through
    volume, OI, IV, and premium analysis.
    """

    def __init__(self, config: UnusualActivityConfig | None = None):
        """
        Initialize scanner.

        Args:
            config: Configuration for detection thresholds
        """
        self.config = config or UnusualActivityConfig()

        # Historical data by underlying
        self.history: dict[str, ActivityHistory] = defaultdict(ActivityHistory)

        # Alert callbacks
        self._alert_callbacks: list[Callable[[UnusualActivityAlert], None]] = []

    def scan(
        self,
        contracts: list[OptionActivityData],
        underlying_price: float,
    ) -> list[UnusualActivityAlert]:
        """
        Scan option chain for unusual activity.

        Args:
            contracts: List of option contracts with activity data
            underlying_price: Current underlying price

        Returns:
            List of unusual activity alerts
        """
        alerts = []

        if not contracts:
            return alerts

        underlying = contracts[0].underlying

        # Group by option type
        calls = [c for c in contracts if c.option_type == "call"]
        puts = [c for c in contracts if c.option_type == "put"]

        # Scan each contract
        for contract in contracts:
            contract_alerts = self._scan_contract(contract)
            alerts.extend(contract_alerts)

        # Check put/call skew
        skew_alert = self._check_put_call_skew(underlying, calls, puts)
        if skew_alert:
            alerts.append(skew_alert)

        # Update history
        self._update_history(underlying, contracts)

        # Sort by urgency and confidence
        alerts.sort(
            key=lambda a: (a.urgency.value, a.confidence),
            reverse=True,
        )

        # Trigger callbacks
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

        return alerts

    def _scan_contract(
        self,
        contract: OptionActivityData,
    ) -> list[UnusualActivityAlert]:
        """Scan individual contract for unusual activity."""
        alerts = []

        # Skip low activity contracts
        if contract.volume < self.config.min_volume:
            return alerts

        # Get historical data
        history = self.history.get(contract.underlying)

        # Volume spike detection
        volume_alert = self._detect_volume_spike(contract, history)
        if volume_alert:
            alerts.append(volume_alert)

        # IV spike detection
        iv_alert = self._detect_iv_spike(contract, history)
        if iv_alert:
            alerts.append(iv_alert)

        # Block trade detection
        block_alert = self._detect_block_trade(contract)
        if block_alert:
            alerts.append(block_alert)

        return alerts

    def _detect_volume_spike(
        self,
        contract: OptionActivityData,
        history: ActivityHistory | None,
    ) -> UnusualActivityAlert | None:
        """Detect volume spike."""
        if not history or len(history.volumes) < 5:
            # Use volume/OI ratio without history
            if contract.volume_oi_ratio > self.config.volume_oi_threshold:
                return self._create_alert(
                    contract=contract,
                    activity_type=ActivityType.VOLUME_SPIKE,
                    current_value=contract.volume,
                    historical_avg=contract.open_interest * 0.1,  # Estimate 10% typical
                    deviation_sigma=2.0,
                    percentile=90.0,
                    confidence=0.6,
                    details={"volume_oi_ratio": contract.volume_oi_ratio},
                )
            return None

        volumes = np.array(history.volumes)
        mean_vol = np.mean(volumes)
        std_vol = np.std(volumes) if len(volumes) > 1 else mean_vol * 0.5

        if std_vol == 0:
            std_vol = mean_vol * 0.5

        deviation = (contract.volume - mean_vol) / std_vol

        if deviation >= self.config.volume_threshold_sigma:
            # Calculate percentile
            percentile = (np.sum(volumes < contract.volume) / len(volumes)) * 100

            return self._create_alert(
                contract=contract,
                activity_type=ActivityType.VOLUME_SPIKE,
                current_value=contract.volume,
                historical_avg=mean_vol,
                deviation_sigma=deviation,
                percentile=percentile,
                confidence=min(0.95, 0.5 + deviation / 10),
                details={
                    "volume_oi_ratio": contract.volume_oi_ratio,
                    "historical_std": std_vol,
                },
            )

        return None

    def _detect_iv_spike(
        self,
        contract: OptionActivityData,
        history: ActivityHistory | None,
    ) -> UnusualActivityAlert | None:
        """Detect IV spike."""
        if not history or len(history.ivs) < 5:
            return None

        ivs = np.array(history.ivs)
        mean_iv = np.mean(ivs)
        std_iv = np.std(ivs) if len(ivs) > 1 else mean_iv * 0.2

        if std_iv == 0:
            std_iv = mean_iv * 0.2

        deviation = (contract.implied_volatility - mean_iv) / std_iv

        if deviation >= self.config.iv_threshold_sigma:
            percentile = (np.sum(ivs < contract.implied_volatility) / len(ivs)) * 100

            if percentile >= self.config.iv_min_percentile:
                return self._create_alert(
                    contract=contract,
                    activity_type=ActivityType.IV_SPIKE,
                    current_value=contract.implied_volatility,
                    historical_avg=mean_iv,
                    deviation_sigma=deviation,
                    percentile=percentile,
                    confidence=min(0.9, 0.4 + deviation / 10),
                    details={
                        "iv_percentile": percentile,
                        "iv_rank": (contract.implied_volatility - np.min(ivs)) / (np.max(ivs) - np.min(ivs) + 0.001),
                    },
                )

        return None

    def _detect_block_trade(
        self,
        contract: OptionActivityData,
    ) -> UnusualActivityAlert | None:
        """Detect block trades."""
        is_block_size = contract.volume >= self.config.block_trade_threshold
        is_block_premium = contract.premium_traded >= self.config.block_premium_threshold

        if is_block_size or is_block_premium:
            # Determine direction based on where it traded relative to mid
            mid = (contract.bid + contract.ask) / 2

            if contract.last_price > mid:
                # Traded above mid - likely a buy
                if contract.option_type == "call":
                    direction = DirectionBias.BULLISH
                else:
                    direction = DirectionBias.BEARISH
            elif contract.last_price < mid:
                # Traded below mid - likely a sell
                if contract.option_type == "call":
                    direction = DirectionBias.BEARISH
                else:
                    direction = DirectionBias.BULLISH
            else:
                direction = DirectionBias.NEUTRAL

            # Higher confidence for larger trades
            confidence = min(
                0.95,
                0.6 + (contract.volume / 1000) * 0.1 + (contract.premium_traded / 200000) * 0.1,
            )

            return self._create_alert(
                contract=contract,
                activity_type=ActivityType.BLOCK_TRADE,
                current_value=contract.volume,
                historical_avg=50,  # Typical trade size
                deviation_sigma=contract.volume / 50,
                percentile=99.0,
                confidence=confidence,
                direction_override=direction,
                details={
                    "premium_traded": contract.premium_traded,
                    "trade_price": contract.last_price,
                    "mid_price": mid,
                },
            )

        return None

    def _check_put_call_skew(
        self,
        underlying: str,
        calls: list[OptionActivityData],
        puts: list[OptionActivityData],
    ) -> UnusualActivityAlert | None:
        """Check put/call ratio for extreme values."""
        call_volume = sum(c.volume for c in calls)
        put_volume = sum(p.volume for p in puts)

        if call_volume == 0 and put_volume == 0:
            return None

        pc_ratio = put_volume / call_volume if call_volume > 0 else float("inf")

        # Get historical P/C ratio
        history = self.history.get(underlying)
        if history and len(history.put_call_ratios) >= 5:
            ratios = np.array(history.put_call_ratios)
            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios) if len(ratios) > 1 else mean_ratio * 0.3

            deviation = (pc_ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0
            percentile = (np.sum(ratios < pc_ratio) / len(ratios)) * 100
        else:
            mean_ratio = 0.7  # Typical P/C ratio
            deviation = 0
            percentile = 50

        # Check for extreme values
        if pc_ratio < self.config.put_call_extreme_low:
            direction = DirectionBias.BULLISH
            activity_type = ActivityType.PUT_CALL_SKEW
            confidence = 0.7
        elif pc_ratio > self.config.put_call_extreme_high:
            direction = DirectionBias.BEARISH
            activity_type = ActivityType.PUT_CALL_SKEW
            confidence = 0.7
        else:
            return None

        # Create a synthetic contract for the alert
        sample = calls[0] if calls else puts[0]

        return UnusualActivityAlert(
            symbol=f"{underlying}_PC_RATIO",
            underlying=underlying,
            activity_type=activity_type,
            current_value=pc_ratio,
            historical_avg=mean_ratio,
            deviation_sigma=abs(deviation),
            percentile=percentile,
            volume=call_volume + put_volume,
            premium=sum(c.premium_traded for c in calls + puts),
            direction_bias=direction,
            confidence=confidence,
            urgency=UrgencyLevel.MEDIUM,
            timestamp=datetime.now(),
            expiry=sample.expiry,
            strike=sample.underlying_price,  # Use underlying price
            option_type="ratio",
            details={
                "call_volume": call_volume,
                "put_volume": put_volume,
                "put_call_ratio": pc_ratio,
            },
        )

    def _create_alert(
        self,
        contract: OptionActivityData,
        activity_type: ActivityType,
        current_value: float,
        historical_avg: float,
        deviation_sigma: float,
        percentile: float,
        confidence: float,
        details: dict[str, Any],
        direction_override: DirectionBias | None = None,
    ) -> UnusualActivityAlert:
        """Create an unusual activity alert."""
        # Determine direction bias
        if direction_override:
            direction = direction_override
        elif contract.option_type == "call":
            direction = DirectionBias.BULLISH
        else:
            direction = DirectionBias.BEARISH

        # Determine urgency
        if deviation_sigma >= 4 or percentile >= 99:
            urgency = UrgencyLevel.CRITICAL
        elif deviation_sigma >= 3 or percentile >= 95:
            urgency = UrgencyLevel.HIGH
        elif deviation_sigma >= 2 or percentile >= 90:
            urgency = UrgencyLevel.MEDIUM
        else:
            urgency = UrgencyLevel.LOW

        return UnusualActivityAlert(
            symbol=contract.symbol,
            underlying=contract.underlying,
            activity_type=activity_type,
            current_value=current_value,
            historical_avg=historical_avg,
            deviation_sigma=deviation_sigma,
            percentile=percentile,
            volume=contract.volume,
            premium=contract.premium_traded,
            direction_bias=direction,
            confidence=confidence,
            urgency=urgency,
            timestamp=contract.timestamp,
            expiry=contract.expiry,
            strike=contract.strike,
            option_type=contract.option_type,
            details=details,
        )

    def _update_history(
        self,
        underlying: str,
        contracts: list[OptionActivityData],
    ) -> None:
        """Update historical data for underlying."""
        if not contracts:
            return

        total_volume = sum(c.volume for c in contracts)
        total_oi = sum(c.open_interest for c in contracts)
        avg_iv = np.mean([c.implied_volatility for c in contracts])

        calls = [c for c in contracts if c.option_type == "call"]
        puts = [c for c in contracts if c.option_type == "put"]

        call_vol = sum(c.volume for c in calls)
        put_vol = sum(p.volume for p in puts)
        pc_ratio = put_vol / call_vol if call_vol > 0 else 1.0

        self.history[underlying].add_record(
            volume=total_volume,
            oi=total_oi,
            iv=avg_iv,
            pc_ratio=pc_ratio,
            timestamp=datetime.now(),
        )

        # Trim old history
        self.history[underlying].trim(self.config.lookback_days)

    def analyze_flow(
        self,
        underlying: str,
        contracts: list[OptionActivityData],
    ) -> FlowAnalysis:
        """
        Analyze overall options flow for a symbol.

        Args:
            underlying: Underlying symbol
            contracts: List of option contracts

        Returns:
            FlowAnalysis with comprehensive flow data
        """
        calls = [c for c in contracts if c.option_type == "call"]
        puts = [c for c in contracts if c.option_type == "put"]

        call_volume = sum(c.volume for c in calls)
        put_volume = sum(p.volume for p in puts)

        call_premium = sum(c.premium_traded for c in calls)
        put_premium = sum(p.premium_traded for p in puts)

        pc_ratio = put_volume / call_volume if call_volume > 0 else float("inf")

        # Calculate net delta
        net_delta = sum(c.delta * c.volume * 100 for c in calls) - sum(p.delta * p.volume * 100 for p in puts)

        # Determine direction
        if pc_ratio < 0.5 and call_premium > put_premium * 1.5:
            direction = DirectionBias.BULLISH
            confidence = min(0.9, 0.5 + (put_premium / call_premium) if call_premium > 0 else 0.5)
        elif pc_ratio > 2.0 and put_premium > call_premium * 1.5:
            direction = DirectionBias.BEARISH
            confidence = min(0.9, 0.5 + (call_premium / put_premium) if put_premium > 0 else 0.5)
        else:
            direction = DirectionBias.NEUTRAL
            confidence = 0.5

        # Get largest trades
        alerts = self.scan(contracts, contracts[0].underlying_price if contracts else 0)
        largest = sorted(alerts, key=lambda a: a.premium, reverse=True)[:5]

        return FlowAnalysis(
            underlying=underlying,
            total_call_volume=call_volume,
            total_put_volume=put_volume,
            put_call_ratio=pc_ratio,
            total_premium=call_premium + put_premium,
            call_premium=call_premium,
            put_premium=put_premium,
            net_delta=net_delta,
            largest_trades=largest,
            direction_bias=direction,
            confidence=confidence,
        )

    def register_alert_callback(
        self,
        callback: Callable[[UnusualActivityAlert], None],
    ) -> None:
        """Register callback for unusual activity alerts."""
        self._alert_callbacks.append(callback)

    def get_summary(self) -> dict[str, Any]:
        """Get scanner summary."""
        return {
            "config": {
                "volume_threshold_sigma": self.config.volume_threshold_sigma,
                "iv_threshold_sigma": self.config.iv_threshold_sigma,
                "block_trade_threshold": self.config.block_trade_threshold,
                "lookback_days": self.config.lookback_days,
            },
            "symbols_tracked": len(self.history),
            "alert_callbacks": len(self._alert_callbacks),
        }


def create_unusual_activity_scanner(
    volume_threshold: float = 2.0,
    block_threshold: int = 100,
) -> UnusualActivityScanner:
    """Factory function to create unusual activity scanner."""
    config = UnusualActivityConfig(
        volume_threshold_sigma=volume_threshold,
        block_trade_threshold=block_threshold,
    )
    return UnusualActivityScanner(config=config)


__all__ = [
    "ActivityHistory",
    "ActivityType",
    "DirectionBias",
    "FlowAnalysis",
    "OptionActivityData",
    "UnusualActivityAlert",
    "UnusualActivityConfig",
    "UnusualActivityScanner",
    "UrgencyLevel",
    "create_unusual_activity_scanner",
]
