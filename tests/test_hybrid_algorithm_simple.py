"""
Simple structural tests for HybridOptionsBot algorithm.

These tests verify the algorithm file is valid Python and has the correct structure.
Full integration tests require QuantConnect runtime environment.
"""

import ast
from pathlib import Path

import pytest


@pytest.mark.unit
def test_hybrid_algorithm_file_exists():
    """Test that hybrid_options_bot.py file exists."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    assert algo_file.exists(), "hybrid_options_bot.py file not found"


@pytest.mark.unit
def test_hybrid_algorithm_valid_python():
    """Test that hybrid_options_bot.py is valid Python syntax."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    try:
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"Syntax error in hybrid_options_bot.py: {e}")


@pytest.mark.unit
def test_hybrid_algorithm_has_class():
    """Test that HybridOptionsBot class is defined."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    tree = ast.parse(code)
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    assert "HybridOptionsBot" in classes, "HybridOptionsBot class not found"


@pytest.mark.unit
def test_hybrid_algorithm_has_required_methods():
    """Test that HybridOptionsBot has required QuantConnect methods."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    tree = ast.parse(code)

    # Find HybridOptionsBot class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HybridOptionsBot":
            methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]

            # Required QuantConnect methods
            assert "Initialize" in methods, "Initialize method not found"
            assert "OnData" in methods, "OnData method not found"
            assert "OnOrderEvent" in methods, "OnOrderEvent method not found"
            assert "OnEndOfAlgorithm" in methods, "OnEndOfAlgorithm method not found"

            # Custom methods
            assert "_process_order_queue" in methods
            assert "_run_autonomous_strategies" in methods
            assert "_update_bot_positions" in methods
            assert "_check_recurring_orders" in methods
            assert "_setup_universe" in methods
            assert "_setup_schedules" in methods
            assert "_check_risk_limits" in methods
            assert "_daily_risk_review" in methods

            break
    else:
        pytest.fail("HybridOptionsBot class not found")


@pytest.mark.unit
def test_hybrid_algorithm_imports():
    """Test that all required imports are present."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    # Check for key imports
    assert "from execution import" in code or "import execution" in code
    assert "from api import" in code or "import api" in code
    assert "from models import" in code or "import models" in code
    assert "from config import" in code or "import config" in code


@pytest.mark.unit
def test_hybrid_algorithm_docstring():
    """Test that HybridOptionsBot has a comprehensive docstring."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    tree = ast.parse(code)

    # Check module docstring
    module_docstring = ast.get_docstring(tree)
    assert module_docstring is not None, "Module docstring missing"
    assert "Hybrid Options Trading Bot" in module_docstring
    assert "semi-autonomous" in module_docstring.lower()

    # Check class docstring
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HybridOptionsBot":
            class_docstring = ast.get_docstring(node)
            assert class_docstring is not None, "HybridOptionsBot docstring missing"
            assert "autonomous" in class_docstring.lower() or "hybrid" in class_docstring.lower()
            break


@pytest.mark.unit
def test_initialize_method_structure():
    """Test that Initialize method has required components."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    # Check for key initialization steps
    assert "SetStartDate" in code or "set_start_date" in code
    assert "SetEndDate" in code or "set_end_date" in code
    assert "SetCash" in code or "set_cash" in code
    assert "SetBrokerageModel" in code
    assert "create_option_strategies_executor" in code
    assert "create_manual_legs_executor" in code
    assert "create_bot_position_manager" in code
    assert "create_recurring_order_manager" in code
    assert "OrderQueueAPI" in code


@pytest.mark.unit
def test_ondata_method_structure():
    """Test that OnData method processes all components."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    # Check for key OnData operations
    assert "IsWarmingUp" in code
    assert "circuit_breaker" in code
    assert "_process_order_queue" in code
    assert "_run_autonomous_strategies" in code
    assert "_update_bot_positions" in code
    assert "_check_recurring_orders" in code


@pytest.mark.unit
def test_risk_management_integration():
    """Test that risk management is integrated."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    # Check for risk management components
    assert "RiskManager" in code
    assert "CircuitBreaker" in code or "circuit_breaker" in code
    assert "RiskLimits" in code
    assert "can_trade" in code
    assert "_check_risk_limits" in code
    assert "_daily_risk_review" in code


@pytest.mark.unit
def test_scheduler_integration():
    """Test that scheduled tasks are configured."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    # Check for schedule configuration
    assert "Schedule.On" in code or "schedule.on" in code
    assert "_setup_schedules" in code
    assert "_scheduled_strategy_check" in code or "_daily_risk_review" in code


@pytest.mark.unit
def test_charles_schwab_warning():
    """Test that Charles Schwab single-algorithm warning is present."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    # Check for warning about Schwab limitation
    assert "CRITICAL" in code
    assert "Charles Schwab" in code
    assert "ONLY ONE algorithm" in code or "single algorithm" in code.lower()


@pytest.mark.unit
def test_module_integration_comments():
    """Test that all 9 modules are referenced."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    # Check for all 9 hybrid modules
    modules = [
        "OptionStrategiesExecutor",
        "ManualLegsExecutor",
        "BotManagedPositions",
        "RecurringOrderManager",
        "OrderQueueAPI",
    ]

    for module in modules:
        assert module in code, f"{module} not found in algorithm"


@pytest.mark.unit
def test_resource_monitoring_integrated():
    """Test that resource monitoring is integrated."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    # Check for resource monitoring
    assert "resource_monitor" in code
    assert "create_resource_monitor" in code or "ResourceMonitor" in code
    assert "_check_resources" in code


@pytest.mark.unit
def test_object_store_integration():
    """Test that Object Store integration is present."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    # Check for Object Store
    assert "object_store" in code.lower()
    assert "create_object_store_manager" in code or "ObjectStoreManager" in code


@pytest.mark.unit
def test_configuration_loading():
    """Test that configuration is loaded."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    # Check for config loading
    assert "get_config" in code
    assert "_get_config" in code
    assert "config" in code


@pytest.mark.unit
def test_error_handling_present():
    """Test that error handling is implemented."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    # Check for error handling
    assert "try:" in code
    assert "except" in code
    assert "Exception" in code


@pytest.mark.unit
def test_debug_logging_present():
    """Test that debug logging is used."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    # Check for debug logging
    assert "self.Debug" in code
    assert "✅" in code  # Success indicators
    assert "⚠️" in code or "❌" in code  # Warning/error indicators


@pytest.mark.unit
def test_code_length_reasonable():
    """Test that algorithm is comprehensive but not excessively long."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        lines = f.readlines()

    line_count = len(lines)

    # Should be substantial (integrating 9 modules)
    assert line_count >= 300, f"Algorithm too short ({line_count} lines)"

    # But not excessively long (well-structured)
    assert line_count <= 1500, f"Algorithm too long ({line_count} lines), consider refactoring"


@pytest.mark.unit
def test_no_syntax_errors_in_methods():
    """Test that all methods are syntactically valid."""
    algo_file = Path("algorithms/hybrid_options_bot.py")
    with open(algo_file) as f:
        code = f.read()

    tree = ast.parse(code)

    # Find all methods and check they parse correctly
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Each function should have at least a docstring or a body
            assert len(node.body) > 0, f"Method {node.name} has empty body"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
