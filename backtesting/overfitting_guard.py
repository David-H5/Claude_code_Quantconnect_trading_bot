"""
Overfitting Guard Module

UPGRADE-015 Phase 9: Backtesting Robustness

Provides overfitting detection for backtested strategies:
- Degrees of freedom analysis
- Out-of-sample validation
- Statistical significance tests
- Multiple comparison correction

Features:
- Sharpe ratio haircut calculation
- Parameter count analysis
- Deflated Sharpe Ratio
- CSCV (Combinatorial Symmetric Cross-Validation)
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats


class OverfitRisk(Enum):
    """Overfitting risk classification."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DeflatedSharpeResult:
    """Results from Deflated Sharpe Ratio analysis."""

    original_sharpe: float
    deflated_sharpe: float
    probability_overfit: float
    haircut_pct: float
    min_track_record: int  # Minimum months needed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_sharpe": self.original_sharpe,
            "deflated_sharpe": self.deflated_sharpe,
            "probability_overfit": self.probability_overfit,
            "haircut_pct": self.haircut_pct,
            "min_track_record_months": self.min_track_record,
        }


@dataclass
class OverfittingResult:
    """Complete overfitting analysis results."""

    # Risk assessment
    overfit_risk: OverfitRisk
    overfit_probability: float
    confidence_level: float

    # Sharpe analysis
    original_sharpe: float
    adjusted_sharpe: float
    sharpe_haircut: float

    # Statistical tests
    deflated_sharpe: DeflatedSharpeResult | None
    in_sample_r_squared: float
    out_sample_r_squared: float

    # Degrees of freedom analysis
    num_parameters: int
    data_points: int
    degrees_of_freedom_ratio: float

    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overfit_risk": self.overfit_risk.value,
            "overfit_probability": self.overfit_probability,
            "original_sharpe": self.original_sharpe,
            "adjusted_sharpe": self.adjusted_sharpe,
            "sharpe_haircut": self.sharpe_haircut,
            "degrees_of_freedom_ratio": self.degrees_of_freedom_ratio,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
        }


@dataclass
class OverfitConfig:
    """Configuration for overfitting detection."""

    # Risk thresholds
    low_risk_prob: float = 0.20
    moderate_risk_prob: float = 0.40
    high_risk_prob: float = 0.60

    # Minimum requirements
    min_data_points: int = 252  # 1 year
    min_dof_ratio: int = 10  # Data points per parameter
    min_out_sample_pct: float = 0.30  # 30% out of sample

    # Statistical thresholds
    significance_level: float = 0.05
    num_trials_adjustment: int = 100  # For multiple comparison

    # Sharpe haircut
    base_haircut_pct: float = 0.10  # 10% base haircut


class OverfittingGuard:
    """Overfitting detection and analysis."""

    def __init__(
        self,
        config: OverfitConfig | None = None,
    ):
        """
        Initialize overfitting guard.

        Args:
            config: Detection configuration
        """
        self.config = config or OverfitConfig()

    # ==========================================================================
    # Degrees of Freedom Analysis
    # ==========================================================================

    def analyze_degrees_of_freedom(
        self,
        num_parameters: int,
        data_points: int,
        num_rules: int = 0,
    ) -> tuple[float, list[str]]:
        """
        Analyze degrees of freedom and data mining risk.

        Args:
            num_parameters: Number of tunable parameters
            data_points: Number of data points
            num_rules: Number of trading rules

        Returns:
            (risk_score, warnings)
        """
        warnings = []

        # Total degrees of freedom consumed
        total_dof = num_parameters + num_rules
        dof_ratio = data_points / total_dof if total_dof > 0 else float("inf")

        # Check minimum ratio
        if dof_ratio < self.config.min_dof_ratio:
            warnings.append(
                f"Low DOF ratio ({dof_ratio:.1f}), need {self.config.min_dof_ratio} data points per parameter"
            )

        # Check minimum data
        if data_points < self.config.min_data_points:
            warnings.append(f"Insufficient data ({data_points}), need {self.config.min_data_points} points")

        # Risk score (0-1, higher = more risk)
        if dof_ratio >= 50:
            risk_score = 0.1
        elif dof_ratio >= 20:
            risk_score = 0.3
        elif dof_ratio >= 10:
            risk_score = 0.5
        elif dof_ratio >= 5:
            risk_score = 0.7
        else:
            risk_score = 0.9

        return risk_score, warnings

    # ==========================================================================
    # Deflated Sharpe Ratio
    # ==========================================================================

    def calculate_deflated_sharpe(
        self,
        sharpe_ratio: float,
        num_trials: int,
        track_record_months: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> DeflatedSharpeResult:
        """
        Calculate Deflated Sharpe Ratio (DSR) per Bailey & Lopez de Prado.

        Adjusts Sharpe ratio for multiple testing and non-normal returns.

        Args:
            sharpe_ratio: Observed Sharpe ratio
            num_trials: Number of strategy variations tried
            track_record_months: Length of track record in months
            skewness: Return distribution skewness
            kurtosis: Return distribution kurtosis

        Returns:
            DeflatedSharpeResult
        """
        # Expected maximum Sharpe from random trials
        e_max = self._expected_max_sharpe(num_trials)

        # Standard deviation of Sharpe estimate
        sr_std = math.sqrt((1 + (0.5 * sharpe_ratio**2) - (skewness * sharpe_ratio)) / track_record_months)

        if kurtosis != 3:
            # Adjust for non-normal kurtosis
            sr_std *= math.sqrt(1 + (kurtosis - 3) / 4)

        # Deflated Sharpe Ratio
        if sr_std > 0:
            deflated_sr = (sharpe_ratio - e_max) / sr_std
        else:
            deflated_sr = 0.0

        # Probability that true Sharpe > 0 (accounting for overfitting)
        prob_overfit = 1 - stats.norm.cdf(deflated_sr)

        # Haircut percentage
        if sharpe_ratio > 0:
            haircut = 1 - max(deflated_sr, 0) / sharpe_ratio
        else:
            haircut = 1.0

        # Minimum track record for significance
        min_track = self._minimum_track_record(sharpe_ratio, skewness, kurtosis)

        return DeflatedSharpeResult(
            original_sharpe=sharpe_ratio,
            deflated_sharpe=deflated_sr,
            probability_overfit=prob_overfit,
            haircut_pct=haircut,
            min_track_record=min_track,
        )

    def _expected_max_sharpe(self, num_trials: int) -> float:
        """Calculate expected maximum Sharpe from random trials."""
        if num_trials <= 1:
            return 0.0

        # Euler-Mascheroni constant approximation
        gamma = 0.5772156649

        # Expected max from N(0,1) samples
        e_max = (1 - gamma) * stats.norm.ppf(1 - 1 / num_trials) + gamma * stats.norm.ppf(1 - 1 / (num_trials * math.e))
        return max(e_max, 0)

    def _minimum_track_record(
        self,
        target_sharpe: float,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> int:
        """Calculate minimum track record needed for significance."""
        if target_sharpe <= 0:
            return 9999

        # From Min TRL formula
        z_alpha = stats.norm.ppf(1 - self.config.significance_level)

        numer = 1 + (0.5 * target_sharpe**2) - (skewness * target_sharpe)
        if kurtosis != 3:
            numer *= 1 + (kurtosis - 3) / 4

        min_months = (z_alpha / target_sharpe) ** 2 * numer
        return max(int(math.ceil(min_months)), 12)

    # ==========================================================================
    # In-Sample vs Out-of-Sample Analysis
    # ==========================================================================

    def analyze_in_vs_out_sample(
        self,
        in_sample_returns: list[float],
        out_sample_returns: list[float],
    ) -> tuple[float, float, list[str]]:
        """
        Compare in-sample and out-of-sample performance.

        Args:
            in_sample_returns: In-sample returns
            out_sample_returns: Out-of-sample returns

        Returns:
            (in_sample_sharpe, out_sample_sharpe, warnings)
        """
        warnings = []

        # Calculate Sharpe ratios
        in_sample = np.array(in_sample_returns)
        out_sample = np.array(out_sample_returns)

        in_sharpe = np.mean(in_sample) / np.std(in_sample) * np.sqrt(252) if np.std(in_sample) > 0 else 0
        out_sharpe = np.mean(out_sample) / np.std(out_sample) * np.sqrt(252) if np.std(out_sample) > 0 else 0

        # Check degradation
        if in_sharpe > 0:
            degradation = (in_sharpe - out_sharpe) / in_sharpe
            if degradation > 0.50:
                warnings.append(f"Severe performance degradation ({degradation:.0%}) suggests overfitting")
            elif degradation > 0.30:
                warnings.append(f"Moderate performance degradation ({degradation:.0%})")

        # Check out-sample size
        out_pct = len(out_sample) / (len(in_sample) + len(out_sample))
        if out_pct < self.config.min_out_sample_pct:
            warnings.append(f"Out-of-sample too small ({out_pct:.0%}), need {self.config.min_out_sample_pct:.0%}")

        return in_sharpe, out_sharpe, warnings

    # ==========================================================================
    # Sharpe Ratio Haircut
    # ==========================================================================

    def calculate_sharpe_haircut(
        self,
        sharpe_ratio: float,
        num_parameters: int,
        data_points: int,
        num_trials: int = 1,
        track_record_months: int = 36,
    ) -> tuple[float, float]:
        """
        Calculate conservative Sharpe ratio estimate with haircut.

        Args:
            sharpe_ratio: Observed Sharpe ratio
            num_parameters: Number of parameters
            data_points: Number of data points
            num_trials: Number of variations tried
            track_record_months: Track record length

        Returns:
            (adjusted_sharpe, haircut_percentage)
        """
        haircut = self.config.base_haircut_pct

        # Additional haircut for low DOF
        dof_ratio = data_points / num_parameters if num_parameters > 0 else float("inf")
        if dof_ratio < 20:
            haircut += (20 - dof_ratio) / 20 * 0.20  # Up to 20% more

        # Additional haircut for multiple trials
        if num_trials > 1:
            trial_haircut = math.log(num_trials) * 0.05
            haircut += min(trial_haircut, 0.30)

        # Additional haircut for short track record
        if track_record_months < 36:
            haircut += (36 - track_record_months) / 36 * 0.15

        # Cap total haircut
        haircut = min(haircut, 0.70)

        adjusted_sharpe = sharpe_ratio * (1 - haircut)
        return adjusted_sharpe, haircut

    # ==========================================================================
    # Full Analysis
    # ==========================================================================

    def analyze(
        self,
        returns: list[float],
        num_parameters: int,
        num_trials: int = 1,
        in_sample_returns: list[float] | None = None,
        out_sample_returns: list[float] | None = None,
    ) -> OverfittingResult:
        """
        Perform complete overfitting analysis.

        Args:
            returns: Full return series
            num_parameters: Number of tunable parameters
            num_trials: Number of strategy variations tried
            in_sample_returns: Optional in-sample returns
            out_sample_returns: Optional out-of-sample returns

        Returns:
            OverfittingResult
        """
        if not returns:
            return OverfittingResult(
                overfit_risk=OverfitRisk.CRITICAL,
                overfit_probability=1.0,
                confidence_level=0.0,
                original_sharpe=0.0,
                adjusted_sharpe=0.0,
                sharpe_haircut=1.0,
                deflated_sharpe=None,
                in_sample_r_squared=0.0,
                out_sample_r_squared=0.0,
                num_parameters=num_parameters,
                data_points=0,
                degrees_of_freedom_ratio=0.0,
                warnings=["No return data provided"],
            )

        returns_array = np.array(returns)
        data_points = len(returns)
        track_record_months = data_points // 21  # Approximate

        # Calculate base Sharpe
        original_sharpe = (
            np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
        )

        # Calculate skewness and kurtosis
        skewness = float(stats.skew(returns_array))
        kurtosis = float(stats.kurtosis(returns_array) + 3)  # Convert to raw kurtosis

        # DOF analysis
        dof_risk, dof_warnings = self.analyze_degrees_of_freedom(num_parameters, data_points)
        dof_ratio = data_points / num_parameters if num_parameters > 0 else float("inf")

        # Deflated Sharpe
        deflated = self.calculate_deflated_sharpe(original_sharpe, num_trials, track_record_months, skewness, kurtosis)

        # Sharpe haircut
        adjusted_sharpe, haircut = self.calculate_sharpe_haircut(
            original_sharpe, num_parameters, data_points, num_trials, track_record_months
        )

        # In vs out sample analysis
        in_r2, out_r2 = 0.0, 0.0
        warnings = list(dof_warnings)

        if in_sample_returns and out_sample_returns:
            in_sharpe, out_sharpe, oos_warnings = self.analyze_in_vs_out_sample(in_sample_returns, out_sample_returns)
            warnings.extend(oos_warnings)

            # Simple R-squared proxies
            in_r2 = min(in_sharpe / 2, 1.0) if in_sharpe > 0 else 0
            out_r2 = min(out_sharpe / 2, 1.0) if out_sharpe > 0 else 0

        # Calculate overall overfit probability
        overfit_prob = self._calculate_overfit_probability(dof_risk, deflated.probability_overfit, in_r2, out_r2)

        # Classify risk
        overfit_risk = self._classify_risk(overfit_prob)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overfit_risk, dof_ratio, num_trials, track_record_months, deflated
        )

        return OverfittingResult(
            overfit_risk=overfit_risk,
            overfit_probability=overfit_prob,
            confidence_level=1 - overfit_prob,
            original_sharpe=original_sharpe,
            adjusted_sharpe=adjusted_sharpe,
            sharpe_haircut=haircut,
            deflated_sharpe=deflated,
            in_sample_r_squared=in_r2,
            out_sample_r_squared=out_r2,
            num_parameters=num_parameters,
            data_points=data_points,
            degrees_of_freedom_ratio=dof_ratio,
            recommendations=recommendations,
            warnings=warnings,
        )

    def _calculate_overfit_probability(
        self,
        dof_risk: float,
        deflated_prob: float,
        in_r2: float,
        out_r2: float,
    ) -> float:
        """Calculate combined overfitting probability."""
        # Weight different factors
        weights = {
            "dof": 0.30,
            "deflated": 0.40,
            "degradation": 0.30,
        }

        # Degradation component
        if in_r2 > 0:
            degradation_risk = max(0, (in_r2 - out_r2) / in_r2)
        else:
            degradation_risk = 0.0

        combined = (
            weights["dof"] * dof_risk + weights["deflated"] * deflated_prob + weights["degradation"] * degradation_risk
        )

        return min(max(combined, 0), 1)

    def _classify_risk(self, overfit_prob: float) -> OverfitRisk:
        """Classify overfitting risk level."""
        if overfit_prob >= self.config.high_risk_prob:
            return OverfitRisk.CRITICAL
        elif overfit_prob >= self.config.moderate_risk_prob:
            return OverfitRisk.HIGH
        elif overfit_prob >= self.config.low_risk_prob:
            return OverfitRisk.MODERATE
        return OverfitRisk.LOW

    def _generate_recommendations(
        self,
        risk: OverfitRisk,
        dof_ratio: float,
        num_trials: int,
        track_record: int,
        deflated: DeflatedSharpeResult,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if risk in [OverfitRisk.HIGH, OverfitRisk.CRITICAL]:
            recommendations.append("Consider reducing number of parameters")
            recommendations.append("Extend out-of-sample testing period")

        if dof_ratio < 20:
            recommendations.append(f"Need more data (current ratio: {dof_ratio:.1f})")

        if num_trials > 10:
            recommendations.append(f"High trial count ({num_trials}) increases false discovery risk")

        if track_record < deflated.min_track_record:
            recommendations.append(f"Need {deflated.min_track_record} months track record (have {track_record})")

        if deflated.haircut_pct > 0.30:
            recommendations.append(f"Use adjusted Sharpe ({deflated.deflated_sharpe:.2f}) for realistic expectations")

        if not recommendations:
            recommendations.append("Strategy passes basic overfitting checks")

        return recommendations


def create_overfitting_guard(
    min_dof_ratio: int = 10,
    significance_level: float = 0.05,
) -> OverfittingGuard:
    """
    Factory function to create an overfitting guard.

    Args:
        min_dof_ratio: Minimum data points per parameter
        significance_level: Statistical significance level

    Returns:
        Configured OverfittingGuard
    """
    config = OverfitConfig(
        min_dof_ratio=min_dof_ratio,
        significance_level=significance_level,
    )
    return OverfittingGuard(config)
