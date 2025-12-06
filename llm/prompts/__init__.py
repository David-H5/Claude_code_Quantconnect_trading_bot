"""
LLM Prompt Templates Package

This package contains all prompt templates for the multi-agent trading system,
organized by agent role with full version control and performance tracking.

Usage:
    from llm.prompts import get_prompt, get_registry

    # Get active supervisor prompt
    prompt = get_prompt(AgentRole.SUPERVISOR)

    # Get specific version
    prompt = get_prompt(AgentRole.TECHNICAL_ANALYST, version="v1.0")

    # Access registry for advanced operations
    registry = get_registry()
    registry.compare_versions(AgentRole.SUPERVISOR, "v1.0", "v1.1")

QuantConnect Compatible: Yes
"""

# Import core registry components
# Import all prompt template modules (this triggers auto-registration)
from llm.prompts import (
    analyst_prompts,
    risk_prompts,
    supervisor_prompts,
    trader_prompts,
)
from llm.prompts.prompt_registry import (
    AgentRole,
    PromptMetrics,
    PromptRegistry,
    PromptVersion,
    get_prompt,
    get_registry,
    register_prompt,
)


__all__ = [
    # Core types
    "AgentRole",
    "PromptVersion",
    "PromptMetrics",
    "PromptRegistry",
    # Main functions
    "get_registry",
    "get_prompt",
    "register_prompt",
    # Prompt modules
    "supervisor_prompts",
    "analyst_prompts",
    "trader_prompts",
    "risk_prompts",
]


def list_all_prompts() -> dict:
    """
    List all registered prompts by role.

    Returns:
        Dictionary mapping role names to list of available versions
    """
    registry = get_registry()
    result = {}

    for role_key, versions in registry.prompts.items():
        result[role_key] = [
            {
                "version": v.version,
                "model": v.model,
                "description": v.description,
                "active": v.version == registry.active_versions.get(role_key),
                "total_uses": v.metrics.total_uses,
                "avg_confidence": v.metrics.avg_confidence,
            }
            for v in versions
        ]

    return result


def get_active_prompts() -> dict:
    """
    Get all currently active prompts.

    Returns:
        Dictionary mapping role names to active PromptVersion
    """
    registry = get_registry()
    result = {}

    for role in AgentRole:
        prompt = registry.get_prompt(role, version="active")
        if prompt:
            result[role.value] = prompt

    return result


def print_prompt_summary() -> None:
    """Print a summary of all registered prompts."""
    prompts = list_all_prompts()

    print("\n=== Registered Prompt Templates ===\n")

    for role_name, versions in prompts.items():
        print(f"{role_name}:")
        for v in versions:
            status = " [ACTIVE]" if v["active"] else ""
            uses = v["total_uses"]
            conf = v["avg_confidence"]
            print(f"  - {v['version']} ({v['model']}){status}")
            print(f"    {v['description']}")
            if uses > 0:
                print(f"    Uses: {uses}, Avg Confidence: {conf:.2f}")
        print()
