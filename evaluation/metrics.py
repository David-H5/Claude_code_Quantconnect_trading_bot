"""
Performance metrics calculation for agent evaluation.

Includes agent-specific metrics, STOCKBENCH standard metrics, and v6.1 enhanced metrics.
"""

from typing import Any


def calculate_agent_metrics(agent_type: str, test_results: list[Any]) -> dict[str, Any]:
    """
    Calculate agent-specific performance metrics.

    Args:
        agent_type: Type of agent being evaluated
        test_results: List of TestResult objects

    Returns:
        Dict of agent-specific metrics
    """
    if agent_type in ["TechnicalAnalyst", "SentimentAnalyst"]:
        return _calculate_analyst_metrics(test_results)
    elif agent_type in ["ConservativeTrader", "ModerateTrader", "AggressiveTrader"]:
        return _calculate_trader_metrics(test_results, agent_type)
    elif agent_type in ["PositionRiskManager", "PortfolioRiskManager", "CircuitBreakerManager"]:
        return _calculate_risk_manager_metrics(test_results)
    elif agent_type == "Supervisor":
        return _calculate_supervisor_metrics(test_results)
    else:
        return {}


def _calculate_analyst_metrics(test_results: list[Any]) -> dict[str, Any]:
    """Calculate metrics for analyst agents."""
    total = len(test_results)
    if total == 0:
        return {}

    # Signal accuracy
    correct_signals = sum(1 for r in test_results if r.actual_output.get("signal") == r.expected_output.get("signal"))
    accuracy = correct_signals / total

    # False positive rate (incorrect bullish/bearish calls)
    false_positives = sum(
        1
        for r in test_results
        if r.actual_output.get("signal") != "neutral"
        and r.actual_output.get("signal") != r.expected_output.get("signal")
    )
    fp_rate = false_positives / total

    # Confidence calibration (RMSE of confidence errors)
    confidence_errors = [
        abs(r.actual_output.get("confidence", 0.5) - r.expected_output.get("confidence", 0.5)) ** 2
        for r in test_results
        if "confidence" in r.actual_output and "confidence" in r.expected_output
    ]
    confidence_rmse = (sum(confidence_errors) / len(confidence_errors)) ** 0.5 if confidence_errors else 0

    # Out-of-sample validation usage
    oos_validated = sum(1 for r in test_results if r.actual_output.get("out_of_sample_validated", False))
    oos_usage_rate = oos_validated / total

    return {
        "accuracy": accuracy,
        "false_positive_rate": fp_rate,
        "confidence_rmse": confidence_rmse,
        "out_of_sample_usage_rate": oos_usage_rate,
        "pattern_recognition_accuracy": accuracy,  # Alias for analysts
    }


def _calculate_trader_metrics(test_results: list[Any], agent_type: str) -> dict[str, Any]:
    """Calculate metrics for trader agents."""
    total = len(test_results)
    if total == 0:
        return {}

    # Win rate (successful position sizing decisions)
    wins = sum(1 for r in test_results if r.passed and r.category == "success")
    total_trades = sum(1 for r in test_results if r.category in ["success", "edge"])
    win_rate = wins / total_trades if total_trades > 0 else 0

    # Kelly calculation accuracy
    kelly_errors = []
    for r in test_results:
        if "position_size_pct" in r.actual_output and "position_size_pct" in r.expected_output:
            expected_size = r.expected_output["position_size_pct"]
            actual_size = r.actual_output["position_size_pct"]
            # Allow 10% tolerance in position sizing
            error = abs(actual_size - expected_size) / expected_size if expected_size > 0 else 0
            kelly_errors.append(error)

    kelly_accuracy = 1.0 - (sum(kelly_errors) / len(kelly_errors)) if kelly_errors else 0

    # Out-of-sample adjustment usage
    oos_adjusted = sum(1 for r in test_results if r.actual_output.get("out_of_sample_adjusted", False))
    oos_usage_rate = oos_adjusted / total

    # Target win rates by trader type
    target_win_rates = {
        "ConservativeTrader": 0.65,
        "ModerateTrader": 0.60,
        "AggressiveTrader": 0.55,
    }
    target_win_rate = target_win_rates.get(agent_type, 0.60)

    return {
        "win_rate": win_rate,
        "kelly_calculation_accuracy": kelly_accuracy,
        "out_of_sample_adjustment_rate": oos_usage_rate,
        "target_win_rate": target_win_rate,
        "meets_target": win_rate >= target_win_rate,
    }


def _calculate_risk_manager_metrics(test_results: list[Any]) -> dict[str, Any]:
    """Calculate metrics for risk manager agents."""
    total = len(test_results)
    if total == 0:
        return {}

    # Zero violations rate (no limit breaches)
    zero_violations = sum(1 for r in test_results if r.actual_output.get("violations", []) == [])
    zero_violations_rate = zero_violations / total

    # Veto accuracy (correct veto decisions)
    veto_correct = sum(
        1 for r in test_results if r.actual_output.get("veto_triggered") == r.expected_output.get("veto_triggered")
    )
    veto_accuracy = veto_correct / total

    # Stop loss effectiveness (properly triggered stops)
    stop_loss_triggered = sum(1 for r in test_results if r.actual_output.get("stop_loss_triggered", False))
    stop_loss_expected = sum(1 for r in test_results if r.expected_output.get("stop_loss_triggered", False))
    stop_loss_effectiveness = stop_loss_triggered / stop_loss_expected if stop_loss_expected > 0 else 1.0

    # Greeks within bounds (for position risk manager)
    greeks_violations = sum(1 for r in test_results if r.actual_output.get("greeks_exceeded", False))
    greeks_compliance_rate = 1.0 - (greeks_violations / total)

    # Predictive circuit breaker effectiveness (for circuit breaker manager)
    preventive_actions = sum(1 for r in test_results if r.actual_output.get("preventive_action_taken", False))
    preventive_expected = sum(1 for r in test_results if r.expected_output.get("preventive_action_expected", False))
    prevention_rate = preventive_actions / preventive_expected if preventive_expected > 0 else 0

    return {
        "zero_violations_rate": zero_violations_rate,
        "veto_accuracy": veto_accuracy,
        "stop_loss_effectiveness": stop_loss_effectiveness,
        "greeks_compliance_rate": greeks_compliance_rate,
        "predictive_prevention_rate": prevention_rate,
    }


def _calculate_supervisor_metrics(test_results: list[Any]) -> dict[str, Any]:
    """Calculate metrics for supervisor agent."""
    total = len(test_results)
    if total == 0:
        return {}

    # Team calibration effectiveness
    calibration_correct = sum(
        1
        for r in test_results
        if r.actual_output.get("team_calibration_triggered", False)
        == r.expected_output.get("team_calibration_triggered", False)
    )
    calibration_effectiveness = calibration_correct / total

    # Token budget adherence
    token_violations = sum(1 for r in test_results if r.actual_output.get("token_budget_exceeded", False))
    token_adherence = 1.0 - (token_violations / total)

    # Task allocation accuracy
    allocation_correct = sum(
        1 for r in test_results if r.actual_output.get("task_allocation") == r.expected_output.get("task_allocation")
    )
    allocation_accuracy = allocation_correct / total

    return {
        "team_calibration_effectiveness": calibration_effectiveness,
        "token_budget_adherence": token_adherence,
        "task_allocation_accuracy": allocation_accuracy,
    }


def calculate_stockbench_metrics(
    returns: list[float], max_drawdown: float, trades: list[dict[str, Any]]
) -> dict[str, float]:
    """
    Calculate STOCKBENCH standard metrics.

    Args:
        returns: List of period returns
        max_drawdown: Maximum drawdown percentage
        trades: List of trade dictionaries with 'pnl' and 'result' keys

    Returns:
        Dict of STOCKBENCH metrics
    """
    if not returns:
        return {}

    # Cumulative return
    cumulative_return = sum(returns)

    # Sortino ratio (downside deviation)
    downside_returns = [r for r in returns if r < 0]
    downside_std = (sum(r**2 for r in downside_returns) / len(downside_returns)) ** 0.5 if downside_returns else 0.0001
    avg_return = sum(returns) / len(returns)
    sortino_ratio = avg_return / downside_std if downside_std > 0 else 0

    # Calmar ratio (return / max drawdown)
    calmar_ratio = cumulative_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    if trades:
        wins = sum(1 for t in trades if t.get("result") == "win")
        win_rate = wins / len(trades)
    else:
        win_rate = 0

    # Profit factor
    if trades:
        gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    else:
        profit_factor = 0

    return {
        "cumulative_return": cumulative_return,
        "maximum_drawdown": max_drawdown,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


def calculate_v6_1_metrics(test_results: list[Any], token_usage: int, team_calibrations: int) -> dict[str, Any]:
    """
    Calculate v6.1 enhanced metrics.

    Args:
        test_results: List of TestResult objects
        token_usage: Total tokens used
        team_calibrations: Number of team calibration events

    Returns:
        Dict of v6.1 enhanced metrics
    """
    total = len(test_results)
    if total == 0:
        return {}

    # Out-of-sample validation accuracy
    oos_cases = [r for r in test_results if r.expected_output.get("out_of_sample_validated")]
    oos_accuracy = sum(1 for r in oos_cases if r.passed) / len(oos_cases) if oos_cases else 0

    # Team calibration effectiveness
    calibration_effectiveness = team_calibrations / (total / 50) if total > 0 else 0

    # Predictive circuit breaker prevention rate
    preventive_cases = [r for r in test_results if r.expected_output.get("preventive_action_expected")]
    prevention_rate = (
        sum(1 for r in preventive_cases if r.actual_output.get("preventive_action_taken")) / len(preventive_cases)
        if preventive_cases
        else 0
    )

    # False positive rate
    false_positives = sum(
        1
        for r in test_results
        if r.category == "failure" and not r.passed  # Should detect failures
    )
    false_positive_rate = false_positives / total

    # Token budget adherence
    token_budget = 50000  # Daily budget
    token_adherence = token_usage <= token_budget

    return {
        "out_of_sample_validation_accuracy": oos_accuracy,
        "team_calibration_effectiveness": calibration_effectiveness,
        "predictive_circuit_breaker_prevention_rate": prevention_rate,
        "false_positive_rate": false_positive_rate,
        "token_budget_adherence": token_adherence,
        "tokens_used": token_usage,
        "token_budget": token_budget,
    }
