"""
Demo Script: Prompt Template System

Demonstrates how to use the prompt template versioning and iteration system.

Usage:
    python scripts/demo_prompt_system.py

Features Demonstrated:
- Registering new prompt versions
- Retrieving active prompts
- Recording usage metrics
- Comparing prompt versions
- A/B testing workflow

QuantConnect Compatible: Yes (standalone script)
"""

from llm.prompts import (
    AgentRole,
    get_prompt,
    get_registry,
    print_prompt_summary,
)


def demo_basic_usage():
    """Demo 1: Basic prompt retrieval."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Prompt Retrieval")
    print("=" * 60)

    # Get active supervisor prompt
    prompt = get_prompt(AgentRole.SUPERVISOR, version="active")

    if prompt:
        print(f"\nAgent: {prompt.role.value}")
        print(f"Version: {prompt.version}")
        print(f"Model: {prompt.model}")
        print(f"Temperature: {prompt.temperature}")
        print(f"Max Tokens: {prompt.max_tokens}")
        print(f"Description: {prompt.description}")
        print("\nPrompt Template (first 200 chars):")
        print(prompt.template[:200] + "...")
    else:
        print("No active prompt found for SUPERVISOR")


def demo_version_listing():
    """Demo 2: List all available versions."""
    print("\n" + "=" * 60)
    print("DEMO 2: List All Prompt Versions")
    print("=" * 60)

    print_prompt_summary()


def demo_usage_tracking():
    """Demo 3: Record usage metrics."""
    print("\n" + "=" * 60)
    print("DEMO 3: Usage Metrics Tracking")
    print("=" * 60)

    registry = get_registry()

    # Simulate some usage
    print("\nSimulating 10 uses of SUPERVISOR v1.0...")

    for i in range(10):
        success = i < 8  # 80% success rate
        response_time = 1200.0 + i * 50  # Increasing response time
        confidence = 0.75 + i * 0.02  # Increasing confidence

        registry.record_usage(
            role=AgentRole.SUPERVISOR,
            version="v1.0",
            success=success,
            response_time_ms=response_time,
            confidence=confidence,
        )

    # Check metrics
    prompt = get_prompt(AgentRole.SUPERVISOR, version="v1.0")
    if prompt:
        print("\nMetrics for SUPERVISOR v1.0:")
        print(f"  Total Uses: {prompt.metrics.total_uses}")
        print(f"  Success Rate: {prompt.metrics.successful_responses}/{prompt.metrics.total_uses}")
        print(f"  Avg Response Time: {prompt.metrics.avg_response_time_ms:.1f}ms")
        print(f"  Avg Confidence: {prompt.metrics.avg_confidence:.3f}")


def demo_version_comparison():
    """Demo 4: Compare two prompt versions."""
    print("\n" + "=" * 60)
    print("DEMO 4: Version Comparison (A/B Testing)")
    print("=" * 60)

    registry = get_registry()

    # Simulate usage for v1.1
    print("\nSimulating 10 uses of SUPERVISOR v1.1...")

    for i in range(10):
        success = i < 9  # 90% success rate (better than v1.0)
        response_time = 1300.0 + i * 40  # Slightly slower
        confidence = 0.80 + i * 0.015  # Similar confidence

        registry.record_usage(
            role=AgentRole.SUPERVISOR,
            version="v1.1",
            success=success,
            response_time_ms=response_time,
            confidence=confidence,
        )

    # Compare versions
    comparison = registry.compare_versions(
        role=AgentRole.SUPERVISOR,
        version1="v1.0",
        version2="v1.1",
    )

    print("\nComparison Results:")
    print(f"  Version 1: {comparison['version1']}")
    print(f"  Version 2: {comparison['version2']}")

    print("\n  Metrics:")
    for metric, data in comparison["metrics_comparison"].items():
        if metric != "total_uses":
            print(f"    {metric}:")
            print(f"      v1.0: {data['v1']}")
            print(f"      v1.1: {data['v2']}")
            print(f"      Winner: {data['winner']}")


def demo_best_version():
    """Demo 5: Get best performing version."""
    print("\n" + "=" * 60)
    print("DEMO 5: Get Best Performing Version")
    print("=" * 60)

    registry = get_registry()

    # Get best by confidence
    best = registry.get_best_version(AgentRole.SUPERVISOR, metric="avg_confidence")

    if best:
        print("\nBest version by avg_confidence:")
        print(f"  Version: {best.version}")
        print(f"  Avg Confidence: {best.metrics.avg_confidence:.3f}")
        print(f"  Total Uses: {best.metrics.total_uses}")


def demo_switching_versions():
    """Demo 6: Switch active version."""
    print("\n" + "=" * 60)
    print("DEMO 6: Switching Active Version")
    print("=" * 60)

    registry = get_registry()

    # Check current active
    current = get_prompt(AgentRole.SUPERVISOR, version="active")
    print(f"\nCurrent active version: {current.version if current else 'None'}")

    # Switch to v1.1
    print("\nSwitching to v1.1...")
    success = registry.set_active(AgentRole.SUPERVISOR, version="v1.1")

    if success:
        new_active = get_prompt(AgentRole.SUPERVISOR, version="active")
        print(f"New active version: {new_active.version if new_active else 'None'}")
        print("Switch successful!")

        # Switch back to v1.0
        print("\nSwitching back to v1.0...")
        registry.set_active(AgentRole.SUPERVISOR, version="v1.0")
        print("Restored original version")
    else:
        print("Failed to switch versions")


def demo_all_roles():
    """Demo 7: Show all registered roles."""
    print("\n" + "=" * 60)
    print("DEMO 7: All Registered Agent Roles")
    print("=" * 60)

    print("\nRegistered roles with active prompts:")

    for role in AgentRole:
        prompt = get_prompt(role, version="active")
        if prompt:
            print(f"\n  {role.value}:")
            print(f"    Version: {prompt.version}")
            print(f"    Model: {prompt.model}")
            print(f"    Temp: {prompt.temperature}")
            print(f"    Description: {prompt.description}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("PROMPT TEMPLATE SYSTEM DEMONSTRATION")
    print("=" * 60)

    demo_basic_usage()
    demo_version_listing()
    demo_usage_tracking()
    demo_version_comparison()
    demo_best_version()
    demo_switching_versions()
    demo_all_roles()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nThe prompt registry has been saved to:")
    print("  llm/prompts/registry.json")
    print("\nYou can inspect this file to see all registered prompts and metrics.")
    print()


if __name__ == "__main__":
    main()
