#!/usr/bin/env python3
"""
Test Compute Node Configuration and Utilities

Validates that all compute node utilities work correctly.

Author: QuantConnect Trading Bot
Date: 2025-11-30
"""

import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from models import create_circuit_breaker
from utils import (
    AlgorithmRequirements,
    NodeOptimizer,
    analyze_algorithm_requirements,
    create_resource_monitor,
)


def test_config_loading():
    """Test that configuration loads correctly."""
    print("\n" + "=" * 60)
    print("TEST: Configuration Loading")
    print("=" * 60)

    config = get_config()
    qc_config = config.get("quantconnect", {})

    # Check compute nodes config
    nodes = qc_config.get("compute_nodes", {})
    assert "backtesting" in nodes, "Missing backtesting node config"
    assert "research" in nodes, "Missing research node config"
    assert "live_trading" in nodes, "Missing live_trading node config"

    # Validate backtesting node
    backtest = nodes["backtesting"]
    assert backtest["model"] == "B8-16", f"Expected B8-16, got {backtest['model']}"
    assert backtest["ram_gb"] == 16, f"Expected 16GB RAM, got {backtest['ram_gb']}"
    assert backtest["cores"] == 8, f"Expected 8 cores, got {backtest['cores']}"

    # Validate research node
    research = nodes["research"]
    assert research["model"] == "R8-16", f"Expected R8-16, got {research['model']}"
    assert research["ram_gb"] == 16, f"Expected 16GB RAM, got {research['ram_gb']}"

    # Validate live trading node
    live = nodes["live_trading"]
    assert live["model"] == "L2-4", f"Expected L2-4, got {live['model']}"
    assert live["ram_gb"] == 4, f"Expected 4GB RAM, got {live['ram_gb']}"
    assert live["colocated"] is True, "Live node should be colocated"

    # Check resource limits
    limits = qc_config.get("resource_limits", {})
    assert "max_option_chains" in limits, "Missing max_option_chains"
    assert "memory_warning_pct" in limits, "Missing memory_warning_pct"

    # Check monitoring config
    monitoring = qc_config.get("monitoring", {})
    assert monitoring.get("enabled") is True, "Monitoring should be enabled"

    print("✓ Configuration loaded successfully")
    print(f"✓ Backtesting: {backtest['model']}")
    print(f"✓ Research: {research['model']}")
    print(f"✓ Live Trading: {live['model']}")
    print("✓ Resource limits configured")
    print("✓ Monitoring enabled")


def test_node_optimizer():
    """Test node optimizer recommendations."""
    print("\n" + "=" * 60)
    print("TEST: Node Optimizer")
    print("=" * 60)

    optimizer = NodeOptimizer()

    # Test with project requirements
    requirements = AlgorithmRequirements(
        num_securities=10,
        num_option_chains=5,
        contracts_per_chain=100,
        max_concurrent_positions=20,
        use_llm_analysis=True,
        use_ml_training=False,
    )

    # Get recommendations
    recommendations = optimizer.recommend_nodes(requirements)

    # Validate backtesting recommendation
    backtest = recommendations["backtesting"]
    assert backtest["node"].model == "B8-16", "Should recommend B8-16 for options"
    print(f"✓ Backtesting: {backtest['node'].model} (${backtest['monthly_cost']}/mo)")
    print(f"  Reason: {backtest['reason']}")

    # Validate research recommendation
    research = recommendations["research"]
    assert research["node"].model == "R8-16", "Should recommend R8-16 for LLM"
    print(f"✓ Research: {research['node'].model} (${research['monthly_cost']}/mo)")
    print(f"  Reason: {research['reason']}")

    # Validate live trading recommendation
    live = recommendations["live_trading"]
    assert live["node"].model == "L2-4", "Should recommend L2-4 for options"
    print(f"✓ Live Trading: {live['node'].model} (${live['monthly_cost']}/mo)")
    print(f"  Reason: {live['reason']}")

    # Validate total cost
    total_cost = recommendations["total_monthly_cost"]
    expected_cost = 28 + 14 + 50  # B8-16 + R8-16 + L2-4
    assert total_cost == expected_cost, f"Expected ${expected_cost}, got ${total_cost}"
    print(f"✓ Total cost: ${total_cost}/month")

    # Test memory estimation
    memory_gb = optimizer.estimate_memory_requirements(requirements)
    print(f"✓ Estimated memory: {memory_gb:.2f}GB")
    assert memory_gb > 3, "Should estimate >3GB for options trading"
    assert memory_gb < 5, "Should estimate <5GB for current config"


def test_resource_monitor():
    """Test resource monitor creation and usage."""
    print("\n" + "=" * 60)
    print("TEST: Resource Monitor")
    print("=" * 60)

    # Create circuit breaker
    breaker = create_circuit_breaker()

    # Create resource monitor
    monitor = create_resource_monitor(
        config={
            "memory_warning_pct": 80,
            "memory_critical_pct": 90,
            "cpu_warning_pct": 75,
            "cpu_critical_pct": 85,
        },
        circuit_breaker=breaker,
    )

    print("✓ Resource monitor created")

    # Test update
    metrics = monitor.update(
        active_securities=10,
        active_positions=5,
        broker_latency_ms=50.0,
    )

    print("✓ Metrics updated:")
    print(f"  Memory: {metrics.memory_pct:.1f}%")
    print(f"  CPU: {metrics.cpu_pct:.1f}%")
    print(f"  Securities: {metrics.active_securities}")
    print(f"  Positions: {metrics.active_positions}")
    print(f"  Latency: {metrics.broker_latency_ms}ms")

    # Test health check
    is_healthy = monitor.is_healthy()
    print(f"✓ Health status: {'Healthy' if is_healthy else 'Warning'}")

    # Test memory estimation
    estimated = monitor.estimate_memory_for_securities(500)
    print(f"✓ Estimated memory for 500 securities: {estimated:.0f}MB")

    # Test can add securities
    can_add = monitor.can_add_securities(100, node_ram_gb=4)
    print(f"✓ Can add 100 securities to L2-4: {can_add}")

    # Test statistics
    stats = monitor.get_statistics()
    if stats:
        print(f"✓ Statistics available: {len(stats)} metrics")


def test_analyze_algorithm():
    """Test algorithm analysis utility."""
    print("\n" + "=" * 60)
    print("TEST: Algorithm Analysis")
    print("=" * 60)

    # Analyze with typical options trading requirements
    recommendations = analyze_algorithm_requirements(
        num_securities=10,
        num_option_chains=5,
        contracts_per_chain=100,
        max_concurrent_positions=20,
        use_llm=True,
        use_ml=False,
    )

    print("✓ Analysis complete")
    print(f"✓ Backtesting: {recommendations['backtesting']['node'].model}")
    print(f"✓ Research: {recommendations['research']['node'].model}")
    print(f"✓ Live Trading: {recommendations['live_trading']['node'].model}")
    print(f"✓ Total cost: ${recommendations['total_monthly_cost']}/month")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("COMPUTE NODE CONFIGURATION TESTS")
    print("=" * 60)

    try:
        test_config_loading()
        test_node_optimizer()
        test_resource_monitor()
        test_analyze_algorithm()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nCompute node configuration is working correctly!")
        print("\nNext steps:")
        print("1. Deploy algorithm to QuantConnect:")
        print("   python scripts/deploy_with_nodes.py algorithms/options_trading_bot.py --analyze-only")
        print("\n2. Read the documentation:")
        print("   docs/infrastructure/COMPUTE_NODES.md")
        print("\n3. Monitor resources in live trading:")
        print("   Check logs/resource_metrics.json")
        print()

        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
