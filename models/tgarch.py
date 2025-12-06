"""
TGARCH (Threshold GARCH) Volatility Model

Implements TGARCH(1,1) model for volatility forecasting with
asymmetric response to positive vs negative returns (leverage effect).

Used by Monte Carlo stress tester for realistic volatility dynamics.

Part of UPGRADE-010 Sprint 4: Risk & Execution.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models.exceptions import DataValidationError, IndicatorError


logger = logging.getLogger(__name__)


@dataclass
class TGARCHParams:
    """
    TGARCH(1,1) model parameters.

    Model: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + gamma * I_{t-1} * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

    Where I_{t-1} = 1 if epsilon_{t-1} < 0 (negative shock), 0 otherwise.
    The gamma parameter captures asymmetric volatility response (leverage effect).
    """

    omega: float  # Constant term (long-run variance component)
    alpha: float  # ARCH coefficient (impact of past squared returns)
    beta: float  # GARCH coefficient (persistence of variance)
    gamma: float  # Asymmetry parameter (leverage effect)

    @property
    def persistence(self) -> float:
        """Volatility persistence (alpha + beta + gamma/2)."""
        return self.alpha + self.beta + self.gamma / 2

    @property
    def long_run_variance(self) -> float:
        """Long-run (unconditional) variance."""
        denom = 1 - self.alpha - self.beta - self.gamma / 2
        return self.omega / denom if denom > 0 else self.omega

    @property
    def long_run_volatility(self) -> float:
        """Long-run (unconditional) volatility."""
        return math.sqrt(self.long_run_variance)

    def is_valid(self) -> tuple[bool, str]:
        """Check if parameters satisfy stationarity conditions."""
        if self.omega <= 0:
            return False, "omega must be positive"
        if self.alpha < 0:
            return False, "alpha must be non-negative"
        if self.beta < 0:
            return False, "beta must be non-negative"
        if self.alpha + self.beta >= 1:
            return False, "alpha + beta must be less than 1 for stationarity"
        if self.persistence >= 1:
            return False, "persistence must be less than 1"
        return True, "valid"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "persistence": self.persistence,
            "long_run_variance": self.long_run_variance,
            "long_run_volatility": self.long_run_volatility,
        }


@dataclass
class TGARCHFitResult:
    """Result of TGARCH model fitting."""

    params: TGARCHParams
    log_likelihood: float
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    num_observations: int
    convergence: bool
    iterations: int
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    conditional_volatility: np.ndarray = field(default_factory=lambda: np.array([]))
    fit_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "params": self.params.to_dict(),
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "num_observations": self.num_observations,
            "convergence": self.convergence,
            "iterations": self.iterations,
            "fit_time_seconds": self.fit_time_seconds,
        }


class TGARCHModel:
    """
    Threshold GARCH(1,1) model for volatility forecasting.

    Captures the leverage effect where negative returns typically
    increase volatility more than positive returns of the same magnitude.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
    ):
        """
        Initialize TGARCH model.

        Args:
            p: ARCH order (default 1)
            q: GARCH order (default 1)
            max_iterations: Maximum iterations for optimization
            tolerance: Convergence tolerance
        """
        self.p = p
        self.q = q
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.params: TGARCHParams | None = None
        self._returns: np.ndarray | None = None
        self._conditional_var: np.ndarray | None = None
        self.is_fitted = False

    def fit(
        self,
        returns: np.ndarray,
        initial_params: TGARCHParams | None = None,
    ) -> TGARCHFitResult:
        """
        Fit TGARCH model to return series.

        Uses simplified Maximum Likelihood Estimation.

        Args:
            returns: Array of returns (not percentages, e.g., 0.01 for 1%)
            initial_params: Optional initial parameter values

        Returns:
            TGARCHFitResult with fitted parameters and diagnostics
        """
        import time

        start_time = time.time()

        self._returns = np.asarray(returns)
        n = len(self._returns)

        if n < 30:
            raise DataValidationError(
                field="returns",
                value=n,
                reason="Need at least 30 observations for TGARCH fitting",
            )

        # Get initial parameters
        if initial_params:
            params = initial_params
        else:
            params = self._get_initial_params(self._returns)

        # Optimize using gradient descent
        best_params = params
        best_ll = self._log_likelihood(params, self._returns)

        for iteration in range(self.max_iterations):
            # Compute gradient numerically
            grad = self._compute_gradient(best_params, self._returns)

            # Update parameters with learning rate decay
            lr = 0.01 / (1 + iteration * 0.01)

            new_omega = max(1e-8, best_params.omega + lr * grad[0])
            new_alpha = max(0, min(0.5, best_params.alpha + lr * grad[1]))
            new_beta = max(0, min(0.95, best_params.beta + lr * grad[2]))
            new_gamma = max(-0.5, min(0.5, best_params.gamma + lr * grad[3]))

            # Ensure stationarity
            if new_alpha + new_beta >= 0.99:
                scale = 0.98 / (new_alpha + new_beta)
                new_alpha *= scale
                new_beta *= scale

            new_params = TGARCHParams(
                omega=new_omega,
                alpha=new_alpha,
                beta=new_beta,
                gamma=new_gamma,
            )

            new_ll = self._log_likelihood(new_params, self._returns)

            if new_ll > best_ll:
                best_params = new_params
                best_ll = new_ll

            # Check convergence
            if iteration > 0 and abs(new_ll - best_ll) < self.tolerance:
                break

        self.params = best_params
        self.is_fitted = True

        # Calculate conditional volatility
        self._conditional_var = self._calculate_conditional_variance(best_params, self._returns)

        # Calculate residuals
        residuals = self._returns / np.sqrt(self._conditional_var)

        # Calculate information criteria
        k = 4  # Number of parameters
        aic = -2 * best_ll + 2 * k
        bic = -2 * best_ll + k * np.log(n)

        fit_time = time.time() - start_time

        return TGARCHFitResult(
            params=best_params,
            log_likelihood=best_ll,
            aic=aic,
            bic=bic,
            num_observations=n,
            convergence=True,
            iterations=iteration + 1,
            residuals=residuals,
            conditional_volatility=np.sqrt(self._conditional_var),
            fit_time_seconds=fit_time,
        )

    def forecast(
        self,
        steps: int = 1,
        return_confidence: bool = False,
    ) -> np.ndarray:
        """
        Forecast volatility for future steps.

        Args:
            steps: Number of steps to forecast
            return_confidence: If True, return confidence intervals

        Returns:
            Array of forecasted volatilities
        """
        if not self.is_fitted or self.params is None:
            raise IndicatorError(
                indicator_name="TGARCH",
                reason="Model must be fitted before forecasting",
            )

        if self._conditional_var is None or len(self._conditional_var) == 0:
            raise IndicatorError(
                indicator_name="TGARCH",
                reason="No conditional variance available",
            )

        forecasts = np.zeros(steps)

        # Get last values
        last_var = self._conditional_var[-1]
        last_return = self._returns[-1] if self._returns is not None else 0
        last_indicator = 1.0 if last_return < 0 else 0.0

        omega = self.params.omega
        alpha = self.params.alpha
        beta = self.params.beta
        gamma = self.params.gamma

        # Forecast variance
        for t in range(steps):
            if t == 0:
                # One-step ahead
                forecasts[t] = (
                    omega + alpha * last_return**2 + gamma * last_indicator * last_return**2 + beta * last_var
                )
            else:
                # Multi-step ahead (converges to long-run variance)
                forecasts[t] = omega + (alpha + beta + gamma / 2) * forecasts[t - 1]

        # Return volatility (sqrt of variance)
        return np.sqrt(forecasts)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate return paths using TGARCH dynamics.

        Args:
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            seed: Random seed for reproducibility

        Returns:
            Tuple of (returns, volatilities) arrays of shape (n_paths, n_steps)
        """
        if not self.is_fitted or self.params is None:
            raise IndicatorError(
                indicator_name="TGARCH",
                reason="Model must be fitted before simulation",
            )

        if seed is not None:
            np.random.seed(seed)

        omega = self.params.omega
        alpha = self.params.alpha
        beta = self.params.beta
        gamma = self.params.gamma

        # Initialize arrays
        returns = np.zeros((n_paths, n_steps))
        variances = np.zeros((n_paths, n_steps))

        # Initial variance (long-run)
        var_0 = self.params.long_run_variance
        variances[:, 0] = var_0

        # Generate standard normal innovations
        z = np.random.standard_normal((n_paths, n_steps))

        for t in range(n_steps):
            if t == 0:
                returns[:, t] = np.sqrt(variances[:, t]) * z[:, t]
            else:
                # TGARCH variance update
                indicator = (returns[:, t - 1] < 0).astype(float)
                variances[:, t] = (
                    omega
                    + alpha * returns[:, t - 1] ** 2
                    + gamma * indicator * returns[:, t - 1] ** 2
                    + beta * variances[:, t - 1]
                )
                returns[:, t] = np.sqrt(variances[:, t]) * z[:, t]

        volatilities = np.sqrt(variances)
        return returns, volatilities

    def _get_initial_params(self, returns: np.ndarray) -> TGARCHParams:
        """Get initial parameter estimates."""
        var = np.var(returns)

        # Reasonable starting values
        omega = var * 0.05  # 5% of variance
        alpha = 0.05
        beta = 0.90
        gamma = 0.05  # Small asymmetry

        return TGARCHParams(
            omega=omega,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    def _log_likelihood(
        self,
        params: TGARCHParams,
        returns: np.ndarray,
    ) -> float:
        """Calculate log-likelihood for given parameters."""
        n = len(returns)
        var = self._calculate_conditional_variance(params, returns)

        # Gaussian log-likelihood
        ll = -0.5 * np.sum(np.log(2 * np.pi * var) + returns**2 / var)

        return ll

    def _calculate_conditional_variance(
        self,
        params: TGARCHParams,
        returns: np.ndarray,
    ) -> np.ndarray:
        """Calculate conditional variance series."""
        n = len(returns)
        var = np.zeros(n)

        # Initialize with unconditional variance
        var[0] = np.var(returns)

        omega = params.omega
        alpha = params.alpha
        beta = params.beta
        gamma = params.gamma

        for t in range(1, n):
            indicator = 1.0 if returns[t - 1] < 0 else 0.0
            var[t] = omega + alpha * returns[t - 1] ** 2 + gamma * indicator * returns[t - 1] ** 2 + beta * var[t - 1]
            # Ensure positive variance
            var[t] = max(1e-10, var[t])

        return var

    def _compute_gradient(
        self,
        params: TGARCHParams,
        returns: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Compute numerical gradient of log-likelihood."""
        grad = np.zeros(4)

        base_ll = self._log_likelihood(params, returns)

        # Gradient for omega
        params_plus = TGARCHParams(params.omega + eps, params.alpha, params.beta, params.gamma)
        grad[0] = (self._log_likelihood(params_plus, returns) - base_ll) / eps

        # Gradient for alpha
        params_plus = TGARCHParams(params.omega, params.alpha + eps, params.beta, params.gamma)
        grad[1] = (self._log_likelihood(params_plus, returns) - base_ll) / eps

        # Gradient for beta
        params_plus = TGARCHParams(params.omega, params.alpha, params.beta + eps, params.gamma)
        grad[2] = (self._log_likelihood(params_plus, returns) - base_ll) / eps

        # Gradient for gamma
        params_plus = TGARCHParams(params.omega, params.alpha, params.beta, params.gamma + eps)
        grad[3] = (self._log_likelihood(params_plus, returns) - base_ll) / eps

        return grad

    def get_summary(self) -> dict[str, Any]:
        """Get model summary."""
        if not self.is_fitted or self.params is None:
            return {"fitted": False}

        return {
            "fitted": True,
            "params": self.params.to_dict(),
            "persistence": self.params.persistence,
            "long_run_volatility": self.params.long_run_volatility,
            "annualized_long_run_vol": self.params.long_run_volatility * np.sqrt(252),
        }


def create_tgarch_model() -> TGARCHModel:
    """Factory function to create TGARCH model."""
    return TGARCHModel()


__all__ = [
    "TGARCHFitResult",
    "TGARCHModel",
    "TGARCHParams",
    "create_tgarch_model",
]
