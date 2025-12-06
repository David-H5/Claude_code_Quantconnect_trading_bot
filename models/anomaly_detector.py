"""
Real-Time Anomaly Detector (UPGRADE-010 Sprint 1)

Detects market regime anomalies using Isolation Forest and integrates
with the circuit breaker for automatic trading halts.

Features:
- Isolation Forest anomaly detection
- Flash crash detection
- Volume spike detection
- Volatility anomaly detection
- Price gap detection
- Circuit breaker integration

Part of UPGRADE-010: Advanced AI Features
Phase: 12 (Real-Time Anomaly Monitoring)

QuantConnect Compatible: Yes
Dependencies: scikit-learn (IsolationForest)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np


# Optional sklearn import with fallback
try:
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    IsolationForest = None

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of market anomalies detected."""

    FLASH_CRASH = "flash_crash"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    PRICE_GAP = "price_gap"
    LIQUIDITY_DROP = "liquidity_drop"
    CORRELATION_BREAK = "correlation_break"
    REGIME_SHIFT = "regime_shift"
    UNKNOWN = "unknown"


class AnomalySeverity(Enum):
    """Severity level of detected anomaly."""

    LOW = "low"  # Informational only
    MEDIUM = "medium"  # Worth monitoring
    HIGH = "high"  # Reduce exposure
    CRITICAL = "critical"  # Halt trading


@dataclass
class AnomalyResult:
    """Result of anomaly detection analysis."""

    is_anomaly: bool
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    score: float  # -1 to 1, lower = more anomalous
    threshold: float
    timestamp: datetime
    feature_values: dict[str, float] = field(default_factory=dict)
    description: str = ""
    recommended_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "score": self.score,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "feature_values": self.feature_values,
            "description": self.description,
            "recommended_action": self.recommended_action,
        }


@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detection thresholds."""

    # Isolation Forest parameters
    contamination: float = 0.01  # Expected proportion of outliers
    n_estimators: int = 100  # Number of trees
    max_samples: int = 256  # Samples per tree
    random_state: int = 42

    # Flash crash detection
    flash_crash_threshold_pct: float = 0.05  # 5% drop in short window
    flash_crash_window_seconds: int = 60  # 1-minute window

    # Volume spike detection
    volume_spike_threshold: float = 3.0  # 3x average volume
    volume_lookback_periods: int = 20  # Periods for average

    # Volatility spike detection
    volatility_spike_threshold: float = 2.5  # 2.5x average volatility
    volatility_lookback_periods: int = 20

    # Price gap detection
    price_gap_threshold_pct: float = 0.03  # 3% gap from previous close

    # General settings
    min_data_points: int = 50  # Minimum data before detection
    enable_circuit_breaker: bool = True

    # Severity thresholds
    medium_score_threshold: float = -0.3
    high_score_threshold: float = -0.5
    critical_score_threshold: float = -0.7

    # Sprint 1.5: Persistence settings
    storage_dir: str = "anomaly_history"
    auto_persist: bool = False
    max_history_size: int = 1000


@dataclass
class MarketDataPoint:
    """Single market data point for anomaly detection."""

    timestamp: datetime
    price: float
    volume: float
    high: float = 0.0
    low: float = 0.0
    open_price: float = 0.0
    bid_ask_spread: float = 0.0

    @property
    def true_range(self) -> float:
        """Calculate true range."""
        if self.high > 0 and self.low > 0:
            return self.high - self.low
        return 0.0


class AnomalyDetector:
    """
    Real-time anomaly detector using Isolation Forest.

    Detects various market anomalies and integrates with circuit breaker
    for automatic trading halts during abnormal market conditions.

    Sprint 1.7: Added Observer pattern for multiple anomaly listeners.

    Usage:
        detector = AnomalyDetector()

        # Add market data
        detector.add_data(MarketDataPoint(
            timestamp=datetime.utcnow(),
            price=450.0,
            volume=1000000,
        ))

        # Check for anomalies
        result = detector.detect()
        if result.is_anomaly:
            if result.severity == AnomalySeverity.CRITICAL:
                circuit_breaker.trip(TripReason.ANOMALY_DETECTED)

        # Observer pattern (Sprint 1.7)
        detector.add_observer(alerting_service.send_anomaly_alert)
        detector.add_observer(continuous_monitor.record_anomaly)
    """

    def __init__(
        self,
        config: AnomalyDetectorConfig | None = None,
        on_anomaly_callback: Callable[[AnomalyResult], None] | None = None,
    ):
        """
        Initialize anomaly detector.

        Args:
            config: Detection configuration
            on_anomaly_callback: Legacy callback when anomaly detected (deprecated, use add_observer)
        """
        self.config = config or AnomalyDetectorConfig()
        self.on_anomaly_callback = on_anomaly_callback

        # Data storage
        self._data_points: list[MarketDataPoint] = []
        self._anomaly_history: list[AnomalyResult] = []

        # Sprint 1.7: Observer pattern for multiple listeners
        self._observers: list[Callable[[AnomalyResult], None]] = []

        # Isolation Forest model
        self._model: Any | None = None
        self._is_fitted: bool = False

        # Feature tracking
        self._returns: list[float] = []
        self._volumes: list[float] = []
        self._volatilities: list[float] = []

        # Initialize model if sklearn available
        if SKLEARN_AVAILABLE:
            self._model = IsolationForest(
                contamination=self.config.contamination,
                n_estimators=self.config.n_estimators,
                max_samples=self.config.max_samples,
                random_state=self.config.random_state,
            )

        # Sprint 1.5: Set up persistence
        if self.config.auto_persist:
            import os

            os.makedirs(self.config.storage_dir, exist_ok=True)

    def add_data(self, data_point: MarketDataPoint) -> None:
        """
        Add a new market data point.

        Args:
            data_point: Market data point to add
        """
        self._data_points.append(data_point)

        # Calculate derived features
        if len(self._data_points) >= 2:
            prev = self._data_points[-2]
            curr = self._data_points[-1]

            # Return
            if prev.price > 0:
                ret = (curr.price - prev.price) / prev.price
                self._returns.append(ret)

            # Volume
            self._volumes.append(curr.volume)

            # Volatility (using true range as proxy)
            if curr.true_range > 0 and curr.price > 0:
                vol = curr.true_range / curr.price
                self._volatilities.append(vol)

        # Limit history
        max_history = 1000
        if len(self._data_points) > max_history:
            self._data_points = self._data_points[-max_history:]
            self._returns = self._returns[-max_history:]
            self._volumes = self._volumes[-max_history:]
            self._volatilities = self._volatilities[-max_history:]

    # =========================================================================
    # Sprint 1.7: Observer Pattern Methods
    # =========================================================================

    def add_observer(self, callback: Callable[[AnomalyResult], None]) -> None:
        """
        Add an observer to be notified when anomalies are detected.

        Sprint 1.7: Observer pattern for multiple listeners.

        Args:
            callback: Function to call with AnomalyResult when anomaly detected

        Example:
            detector.add_observer(alerting_service.send_anomaly_alert)
            detector.add_observer(continuous_monitor.record_anomaly)
        """
        if callback not in self._observers:
            self._observers.append(callback)

    def remove_observer(self, callback: Callable[[AnomalyResult], None]) -> bool:
        """
        Remove an observer.

        Sprint 1.7: Observer pattern for multiple listeners.

        Args:
            callback: The callback to remove

        Returns:
            True if the observer was removed, False if not found
        """
        if callback in self._observers:
            self._observers.remove(callback)
            return True
        return False

    def get_observer_count(self) -> int:
        """
        Get the number of registered observers.

        Returns:
            Number of observers currently registered
        """
        return len(self._observers)

    def _notify_observers(self, result: AnomalyResult) -> None:
        """
        Notify all observers of a detected anomaly.

        Sprint 1.7: Calls all registered observer callbacks.

        Args:
            result: The anomaly result to send to observers
        """
        for observer in self._observers:
            try:
                observer(result)
            except Exception as e:
                logger.warning(f"Observer callback failed: {e}")

    # =========================================================================
    # End Sprint 1.7 Observer Methods
    # =========================================================================

    def detect(self) -> AnomalyResult:
        """
        Run anomaly detection on latest data.

        Returns:
            AnomalyResult with detection results
        """
        if len(self._data_points) < self.config.min_data_points:
            return self._create_no_anomaly_result("Insufficient data")

        # Check specific anomaly types
        flash_crash = self._detect_flash_crash()
        if flash_crash.is_anomaly:
            self._record_anomaly(flash_crash)
            return flash_crash

        volume_spike = self._detect_volume_spike()
        if volume_spike.is_anomaly and volume_spike.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]:
            self._record_anomaly(volume_spike)
            return volume_spike

        volatility_spike = self._detect_volatility_spike()
        if volatility_spike.is_anomaly and volatility_spike.severity in [
            AnomalySeverity.HIGH,
            AnomalySeverity.CRITICAL,
        ]:
            self._record_anomaly(volatility_spike)
            return volatility_spike

        price_gap = self._detect_price_gap()
        if price_gap.is_anomaly:
            self._record_anomaly(price_gap)
            return price_gap

        # Use Isolation Forest for general anomaly detection
        if SKLEARN_AVAILABLE and self._model is not None:
            isolation_result = self._detect_isolation_forest()
            if isolation_result.is_anomaly:
                self._record_anomaly(isolation_result)
                return isolation_result

        return self._create_no_anomaly_result("No anomalies detected")

    def _detect_flash_crash(self) -> AnomalyResult:
        """Detect flash crash (rapid price decline)."""
        if len(self._data_points) < 2:
            return self._create_no_anomaly_result("Insufficient data for flash crash")

        window_start = datetime.utcnow() - timedelta(seconds=self.config.flash_crash_window_seconds)

        # Get data points in window
        window_data = [dp for dp in self._data_points if dp.timestamp >= window_start]

        if len(window_data) < 2:
            return self._create_no_anomaly_result("Insufficient window data")

        # Calculate return over window
        start_price = window_data[0].price
        end_price = window_data[-1].price

        if start_price <= 0:
            return self._create_no_anomaly_result("Invalid price data")

        pct_change = (end_price - start_price) / start_price

        # Check for flash crash (significant drop)
        if pct_change < -self.config.flash_crash_threshold_pct:
            severity = self._calculate_severity_from_score(pct_change * 10)

            return AnomalyResult(
                is_anomaly=True,
                anomaly_type=AnomalyType.FLASH_CRASH,
                severity=severity,
                score=pct_change * 10,  # Scale for comparison
                threshold=-self.config.flash_crash_threshold_pct,
                timestamp=datetime.utcnow(),
                feature_values={
                    "pct_change": pct_change,
                    "window_seconds": self.config.flash_crash_window_seconds,
                    "start_price": start_price,
                    "end_price": end_price,
                },
                description=f"Flash crash detected: {pct_change:.2%} drop in {self.config.flash_crash_window_seconds}s",
                recommended_action="HALT_TRADING",
            )

        return self._create_no_anomaly_result("No flash crash")

    def _detect_volume_spike(self) -> AnomalyResult:
        """Detect unusual volume spike."""
        if len(self._volumes) < self.config.volume_lookback_periods:
            return self._create_no_anomaly_result("Insufficient volume data")

        # Calculate average volume
        lookback = self.config.volume_lookback_periods
        avg_volume = (
            np.mean(self._volumes[-lookback - 1 : -1]) if len(self._volumes) > lookback else np.mean(self._volumes[:-1])
        )
        current_volume = self._volumes[-1]

        if avg_volume <= 0:
            return self._create_no_anomaly_result("Invalid volume data")

        volume_ratio = current_volume / avg_volume

        if volume_ratio > self.config.volume_spike_threshold:
            # Calculate score (higher ratio = lower score = more anomalous)
            score = -min((volume_ratio - 1) / 5, 1.0)  # Normalize to -1 to 0
            severity = self._calculate_severity_from_score(score)

            return AnomalyResult(
                is_anomaly=True,
                anomaly_type=AnomalyType.VOLUME_SPIKE,
                severity=severity,
                score=score,
                threshold=self.config.volume_spike_threshold,
                timestamp=datetime.utcnow(),
                feature_values={
                    "current_volume": current_volume,
                    "average_volume": avg_volume,
                    "volume_ratio": volume_ratio,
                },
                description=f"Volume spike: {volume_ratio:.1f}x average",
                recommended_action="MONITOR" if severity == AnomalySeverity.MEDIUM else "REDUCE_EXPOSURE",
            )

        return self._create_no_anomaly_result("No volume spike")

    def _detect_volatility_spike(self) -> AnomalyResult:
        """Detect unusual volatility spike."""
        if len(self._volatilities) < self.config.volatility_lookback_periods:
            return self._create_no_anomaly_result("Insufficient volatility data")

        lookback = self.config.volatility_lookback_periods
        avg_vol = (
            np.mean(self._volatilities[-lookback - 1 : -1])
            if len(self._volatilities) > lookback
            else np.mean(self._volatilities[:-1])
        )
        current_vol = self._volatilities[-1] if self._volatilities else 0

        if avg_vol <= 0 or current_vol <= 0:
            return self._create_no_anomaly_result("Invalid volatility data")

        vol_ratio = current_vol / avg_vol

        if vol_ratio > self.config.volatility_spike_threshold:
            score = -min((vol_ratio - 1) / 4, 1.0)
            severity = self._calculate_severity_from_score(score)

            return AnomalyResult(
                is_anomaly=True,
                anomaly_type=AnomalyType.VOLATILITY_SPIKE,
                severity=severity,
                score=score,
                threshold=self.config.volatility_spike_threshold,
                timestamp=datetime.utcnow(),
                feature_values={
                    "current_volatility": current_vol,
                    "average_volatility": avg_vol,
                    "volatility_ratio": vol_ratio,
                },
                description=f"Volatility spike: {vol_ratio:.1f}x average",
                recommended_action="REDUCE_EXPOSURE",
            )

        return self._create_no_anomaly_result("No volatility spike")

    def _detect_price_gap(self) -> AnomalyResult:
        """Detect price gap from previous close."""
        if len(self._data_points) < 2:
            return self._create_no_anomaly_result("Insufficient data for gap detection")

        prev = self._data_points[-2]
        curr = self._data_points[-1]

        if prev.price <= 0:
            return self._create_no_anomaly_result("Invalid price data")

        gap_pct = abs(curr.price - prev.price) / prev.price

        if gap_pct > self.config.price_gap_threshold_pct:
            direction = "up" if curr.price > prev.price else "down"
            score = -min(gap_pct * 10, 1.0)
            severity = self._calculate_severity_from_score(score)

            return AnomalyResult(
                is_anomaly=True,
                anomaly_type=AnomalyType.PRICE_GAP,
                severity=severity,
                score=score,
                threshold=self.config.price_gap_threshold_pct,
                timestamp=datetime.utcnow(),
                feature_values={
                    "gap_pct": gap_pct,
                    "direction": direction,
                    "prev_price": prev.price,
                    "curr_price": curr.price,
                },
                description=f"Price gap {direction}: {gap_pct:.2%}",
                recommended_action="MONITOR",
            )

        return self._create_no_anomaly_result("No price gap")

    def _detect_isolation_forest(self) -> AnomalyResult:
        """Use Isolation Forest for general anomaly detection."""
        if not SKLEARN_AVAILABLE or self._model is None:
            return self._create_no_anomaly_result("sklearn not available")

        if len(self._returns) < self.config.min_data_points:
            return self._create_no_anomaly_result("Insufficient data for IF")

        # Prepare features
        features = self._prepare_features()
        if features is None or len(features) < self.config.min_data_points:
            return self._create_no_anomaly_result("Could not prepare features")

        # Fit if not fitted
        if not self._is_fitted:
            try:
                self._model.fit(features)
                self._is_fitted = True
            except Exception as e:
                logger.warning(f"Failed to fit Isolation Forest: {e}")
                return self._create_no_anomaly_result("Model fitting failed")

        # Get score for latest point
        try:
            latest_features = features[-1:].reshape(1, -1)
            score = self._model.score_samples(latest_features)[0]
            prediction = self._model.predict(latest_features)[0]
        except Exception as e:
            logger.warning(f"Failed to predict with Isolation Forest: {e}")
            return self._create_no_anomaly_result("Prediction failed")

        is_anomaly = prediction == -1

        if is_anomaly:
            severity = self._calculate_severity_from_score(score)

            return AnomalyResult(
                is_anomaly=True,
                anomaly_type=AnomalyType.REGIME_SHIFT,
                severity=severity,
                score=score,
                threshold=-self.config.contamination,
                timestamp=datetime.utcnow(),
                feature_values={
                    "isolation_score": score,
                    "features": features[-1].tolist() if len(features) > 0 else [],
                },
                description=f"Isolation Forest anomaly (score: {score:.3f})",
                recommended_action="MONITOR" if severity == AnomalySeverity.LOW else "REDUCE_EXPOSURE",
            )

        return self._create_no_anomaly_result("No IF anomaly")

    def _prepare_features(self) -> np.ndarray | None:
        """Prepare feature matrix for Isolation Forest."""
        min_len = min(
            len(self._returns),
            len(self._volumes),
            len(self._volatilities) if self._volatilities else len(self._returns),
        )

        if min_len < self.config.min_data_points:
            return None

        # Build feature matrix
        features = []

        for i in range(min_len):
            row = [
                self._returns[i] if i < len(self._returns) else 0,
                np.log1p(self._volumes[i]) if i < len(self._volumes) else 0,  # Log volume
            ]
            if self._volatilities and i < len(self._volatilities):
                row.append(self._volatilities[i])
            features.append(row)

        return np.array(features)

    def _calculate_severity_from_score(self, score: float) -> AnomalySeverity:
        """Calculate severity level from anomaly score."""
        if score <= self.config.critical_score_threshold:
            return AnomalySeverity.CRITICAL
        elif score <= self.config.high_score_threshold:
            return AnomalySeverity.HIGH
        elif score <= self.config.medium_score_threshold:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    def _create_no_anomaly_result(self, reason: str) -> AnomalyResult:
        """Create a result indicating no anomaly."""
        return AnomalyResult(
            is_anomaly=False,
            anomaly_type=AnomalyType.UNKNOWN,
            severity=AnomalySeverity.LOW,
            score=0.0,
            threshold=0.0,
            timestamp=datetime.utcnow(),
            description=reason,
            recommended_action="CONTINUE",
        )

    def _record_anomaly(self, result: AnomalyResult) -> None:
        """Record anomaly and trigger callbacks/observers."""
        self._anomaly_history.append(result)

        # Limit history
        max_history = self.config.max_history_size
        if len(self._anomaly_history) > max_history:
            self._anomaly_history = self._anomaly_history[-max_history:]

        # Sprint 1.5: Persist anomaly
        if self.config.auto_persist:
            self._persist_anomaly(result)

        # Legacy callback (deprecated, use add_observer instead)
        if self.on_anomaly_callback:
            try:
                self.on_anomaly_callback(result)
            except Exception as e:
                logger.warning(f"Anomaly callback failed: {e}")

        # Sprint 1.7: Notify all observers
        self._notify_observers(result)

    def get_anomaly_history(
        self,
        limit: int = 50,
        severity_filter: AnomalySeverity | None = None,
    ) -> list[AnomalyResult]:
        """Get recent anomaly history."""
        history = self._anomaly_history

        if severity_filter:
            history = [a for a in history if a.severity == severity_filter]

        return history[-limit:]

    def get_false_positive_rate(self) -> float:
        """
        Estimate false positive rate based on history.

        Returns:
            Estimated false positive rate (0-1)
        """
        if not self._anomaly_history:
            return 0.0

        # Count anomalies by severity
        total = len(self._anomaly_history)
        low_severity = len([a for a in self._anomaly_history if a.severity == AnomalySeverity.LOW])

        # Assume low severity anomalies are potential false positives
        return low_severity / total if total > 0 else 0.0

    def reset(self) -> None:
        """Reset detector state."""
        self._data_points.clear()
        self._returns.clear()
        self._volumes.clear()
        self._volatilities.clear()
        self._is_fitted = False

        if SKLEARN_AVAILABLE and self._model is not None:
            self._model = IsolationForest(
                contamination=self.config.contamination,
                n_estimators=self.config.n_estimators,
                max_samples=self.config.max_samples,
                random_state=self.config.random_state,
            )

    def get_statistics(self) -> dict[str, Any]:
        """Get detector statistics."""
        return {
            "data_points": len(self._data_points),
            "anomalies_detected": len(self._anomaly_history),
            "is_fitted": self._is_fitted,
            "sklearn_available": SKLEARN_AVAILABLE,
            "false_positive_rate_estimate": self.get_false_positive_rate(),
            "anomalies_by_type": self._count_anomalies_by_type(),
            "anomalies_by_severity": self._count_anomalies_by_severity(),
        }

    def _count_anomalies_by_type(self) -> dict[str, int]:
        """Count anomalies by type."""
        counts: dict[str, int] = {}
        for a in self._anomaly_history:
            key = a.anomaly_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _count_anomalies_by_severity(self) -> dict[str, int]:
        """Count anomalies by severity."""
        counts: dict[str, int] = {}
        for a in self._anomaly_history:
            key = a.severity.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    # =========================================================================
    # Sprint 1.5: Persistence Methods
    # =========================================================================

    def _persist_anomaly(self, result: AnomalyResult) -> bool:
        """
        Persist a single anomaly to storage.

        Sprint 1.5: Anomaly history persistence.

        Args:
            result: Anomaly result to persist

        Returns:
            True if persisted successfully
        """
        import hashlib
        import json
        import os

        try:
            # Generate unique filename
            content = f"{result.timestamp.isoformat()}:{result.anomaly_type.value}"
            anomaly_id = hashlib.sha256(content.encode()).hexdigest()[:16]

            filepath = os.path.join(self.config.storage_dir, f"{anomaly_id}.json")
            with open(filepath, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.warning(f"Failed to persist anomaly: {e}")
            return False

    def export_anomaly_audit_trail(
        self,
        filepath: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """
        Export anomaly history as compliance audit trail.

        Sprint 1.5: Anomaly history persistence.

        Args:
            filepath: Output file path
            start_time: Filter start time
            end_time: Filter end time

        Returns:
            Number of anomalies exported
        """
        import json

        history = self._anomaly_history

        # Filter by time
        if start_time:
            history = [a for a in history if a.timestamp >= start_time]
        if end_time:
            history = [a for a in history if a.timestamp <= end_time]

        # Sort by timestamp
        history = sorted(history, key=lambda a: a.timestamp)

        # Build audit trail
        audit_trail = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_anomalies": len(history),
            "filter_start": start_time.isoformat() if start_time else None,
            "filter_end": end_time.isoformat() if end_time else None,
            "anomalies_by_type": self._count_anomalies_by_type(),
            "anomalies_by_severity": self._count_anomalies_by_severity(),
            "false_positive_rate_estimate": self.get_false_positive_rate(),
            "entries": [a.to_dict() for a in history],
        }

        # Write to file
        with open(filepath, "w") as f:
            json.dump(audit_trail, f, indent=2)

        return len(history)

    def load_anomaly_history(self, limit: int = 100) -> int:
        """
        Load anomaly history from storage.

        Sprint 1.5: Anomaly history persistence.

        Args:
            limit: Maximum number of anomalies to load

        Returns:
            Number of anomalies loaded
        """
        import json
        import os

        if not os.path.exists(self.config.storage_dir):
            return 0

        loaded = 0
        files = sorted(os.listdir(self.config.storage_dir))[-limit:]

        for filename in files:
            if not filename.endswith(".json"):
                continue

            try:
                filepath = os.path.join(self.config.storage_dir, filename)
                with open(filepath) as f:
                    data = json.load(f)

                result = AnomalyResult(
                    is_anomaly=data["is_anomaly"],
                    anomaly_type=AnomalyType(data["anomaly_type"]),
                    severity=AnomalySeverity(data["severity"]),
                    score=data["score"],
                    threshold=data["threshold"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    feature_values=data.get("feature_values", {}),
                    description=data.get("description", ""),
                    recommended_action=data.get("recommended_action", ""),
                )
                self._anomaly_history.append(result)
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load anomaly {filename}: {e}")

        return loaded


def create_anomaly_detector(
    config: AnomalyDetectorConfig | None = None,
    on_anomaly_callback: Callable[[AnomalyResult], None] | None = None,
) -> AnomalyDetector:
    """
    Factory function to create an AnomalyDetector.

    Args:
        config: Detection configuration
        on_anomaly_callback: Callback when anomaly detected

    Returns:
        Configured AnomalyDetector
    """
    return AnomalyDetector(
        config=config,
        on_anomaly_callback=on_anomaly_callback,
    )
