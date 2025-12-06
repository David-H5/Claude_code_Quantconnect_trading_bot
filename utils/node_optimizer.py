#!/usr/bin/env python3
"""
Node Optimizer for QuantConnect Compute Nodes

Recommends optimal compute nodes based on algorithm requirements.
Helps select between B8-16 (backtest), R8-16 (research), L2-4 (live).

Author: QuantConnect Trading Bot
Date: 2025-11-30
"""

import logging
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class NodeType(Enum):
    """QuantConnect compute node types."""

    BACKTESTING = "backtesting"
    RESEARCH = "research"
    LIVE_TRADING = "live_trading"


@dataclass
class NodeSpec:
    """Specification for a compute node."""

    model: str
    cores: int
    speed_ghz: float
    ram_gb: float
    gpu: bool
    monthly_cost: int
    colocated: bool = False
    max_latency_ms: float | None = None


@dataclass
class AlgorithmRequirements:
    """Resource requirements for an algorithm."""

    num_securities: int
    num_option_chains: int
    contracts_per_chain: int
    max_concurrent_positions: int
    use_llm_analysis: bool = False
    use_ml_training: bool = False
    tick_data: bool = False
    long_backtest_days: int = 365
    needs_low_latency: bool = True


class NodeOptimizer:
    """
    Optimize QuantConnect compute node selection.

    Analyzes algorithm requirements and recommends the most appropriate
    compute node for backtesting, research, and live trading.

    Example usage:
        optimizer = NodeOptimizer()

        requirements = AlgorithmRequirements(
            num_securities=10,
            num_option_chains=5,
            contracts_per_chain=100,
            max_concurrent_positions=20,
            use_llm_analysis=True,
        )

        # Get recommendations
        recommendations = optimizer.recommend_nodes(requirements)
        print(recommendations["backtesting"])
        print(recommendations["live_trading"])
    """

    def __init__(self):
        """Initialize node optimizer with available node specs."""
        # Define available nodes
        self.nodes = {
            NodeType.BACKTESTING: {
                "B-MICRO": NodeSpec(
                    model="B-MICRO",
                    cores=2,
                    speed_ghz=3.3,
                    ram_gb=8,
                    gpu=False,
                    monthly_cost=0,  # Free but 20s delay
                ),
                "B2-8": NodeSpec(
                    model="B2-8",
                    cores=2,
                    speed_ghz=4.9,
                    ram_gb=8,
                    gpu=False,
                    monthly_cost=14,
                ),
                "B4-12": NodeSpec(
                    model="B4-12",
                    cores=4,
                    speed_ghz=4.9,
                    ram_gb=12,
                    gpu=False,
                    monthly_cost=20,
                ),
                "B8-16": NodeSpec(
                    model="B8-16",
                    cores=8,
                    speed_ghz=4.9,
                    ram_gb=16,
                    gpu=False,
                    monthly_cost=28,
                ),
                "B4-16-GPU": NodeSpec(
                    model="B4-16-GPU",
                    cores=4,
                    speed_ghz=3.0,
                    ram_gb=16,
                    gpu=True,
                    monthly_cost=400,
                ),
            },
            NodeType.RESEARCH: {
                "R1-4": NodeSpec(
                    model="R1-4",
                    cores=1,
                    speed_ghz=2.4,
                    ram_gb=4,
                    gpu=False,
                    monthly_cost=5,
                ),
                "R2-8": NodeSpec(
                    model="R2-8",
                    cores=2,
                    speed_ghz=2.4,
                    ram_gb=8,
                    gpu=False,
                    monthly_cost=10,
                ),
                "R4-12": NodeSpec(
                    model="R4-12",
                    cores=4,
                    speed_ghz=2.4,
                    ram_gb=12,
                    gpu=False,
                    monthly_cost=14,
                ),
                "R8-16": NodeSpec(
                    model="R8-16",
                    cores=8,
                    speed_ghz=2.4,
                    ram_gb=16,
                    gpu=False,
                    monthly_cost=14,
                ),
                "R4-16-GPU": NodeSpec(
                    model="R4-16-GPU",
                    cores=4,
                    speed_ghz=3.0,
                    ram_gb=16,
                    gpu=True,
                    monthly_cost=400,
                ),
            },
            NodeType.LIVE_TRADING: {
                "L-MICRO": NodeSpec(
                    model="L-MICRO",
                    cores=1,
                    speed_ghz=2.6,
                    ram_gb=0.5,
                    gpu=False,
                    monthly_cost=20,
                    colocated=True,
                    max_latency_ms=100,
                ),
                "L1-1": NodeSpec(
                    model="L1-1",
                    cores=1,
                    speed_ghz=2.6,
                    ram_gb=1,
                    gpu=False,
                    monthly_cost=25,
                    colocated=True,
                    max_latency_ms=100,
                ),
                "L1-2": NodeSpec(
                    model="L1-2",
                    cores=1,
                    speed_ghz=2.6,
                    ram_gb=2,
                    gpu=False,
                    monthly_cost=35,
                    colocated=True,
                    max_latency_ms=100,
                ),
                "L2-4": NodeSpec(
                    model="L2-4",
                    cores=2,
                    speed_ghz=2.6,
                    ram_gb=4,
                    gpu=False,
                    monthly_cost=50,
                    colocated=True,
                    max_latency_ms=100,
                ),
                "L8-16-GPU": NodeSpec(
                    model="L8-16-GPU",
                    cores=8,
                    speed_ghz=3.1,
                    ram_gb=16,
                    gpu=True,
                    monthly_cost=400,
                    colocated=True,
                    max_latency_ms=100,
                ),
            },
        }

    def estimate_memory_requirements(self, requirements: AlgorithmRequirements) -> float:
        """
        Estimate memory requirements in GB.

        Args:
            requirements: Algorithm requirements

        Returns:
            Estimated memory in GB
        """
        # Base overhead
        base_mb = 1100

        # Per security overhead
        securities = requirements.num_securities
        security_mb = securities * 5

        # Options overhead (more memory intensive)
        options_contracts = requirements.num_option_chains * requirements.contracts_per_chain
        options_mb = options_contracts * 5

        # LLM analysis overhead (API calls, not local models)
        llm_mb = 50 if requirements.use_llm_analysis else 0

        # ML training overhead
        ml_mb = 500 if requirements.use_ml_training else 0

        # Tick data overhead
        tick_mb = 1000 if requirements.tick_data else 0

        total_mb = base_mb + security_mb + options_mb + llm_mb + ml_mb + tick_mb
        total_gb = total_mb / 1024

        return total_gb

    def recommend_backtesting_node(self, requirements: AlgorithmRequirements) -> dict:
        """
        Recommend optimal backtesting node.

        Args:
            requirements: Algorithm requirements

        Returns:
            Dictionary with recommended node and reasoning
        """
        memory_gb = self.estimate_memory_requirements(requirements)
        options_intensive = requirements.num_option_chains > 0
        long_backtest = requirements.long_backtest_days > 365

        # Scoring logic
        if requirements.use_ml_training and requirements.use_llm_analysis:
            # Need GPU for local ML training
            recommended = self.nodes[NodeType.BACKTESTING]["B4-16-GPU"]
            reason = "GPU required for ML model training"
        elif memory_gb > 12 or options_intensive or long_backtest:
            # Need B8-16
            recommended = self.nodes[NodeType.BACKTESTING]["B8-16"]
            reason = f"High memory requirements ({memory_gb:.1f}GB) or options trading"
        elif memory_gb > 8:
            # Need B4-12
            recommended = self.nodes[NodeType.BACKTESTING]["B4-12"]
            reason = f"Moderate memory requirements ({memory_gb:.1f}GB)"
        elif memory_gb > 0:
            # Can use B2-8
            recommended = self.nodes[NodeType.BACKTESTING]["B2-8"]
            reason = f"Low memory requirements ({memory_gb:.1f}GB)"
        else:
            # Free tier acceptable
            recommended = self.nodes[NodeType.BACKTESTING]["B-MICRO"]
            reason = "Simple algorithm, free tier sufficient (20s delay)"

        return {
            "node": recommended,
            "reason": reason,
            "estimated_memory_gb": memory_gb,
            "monthly_cost": recommended.monthly_cost,
        }

    def recommend_research_node(self, requirements: AlgorithmRequirements) -> dict:
        """
        Recommend optimal research node.

        Args:
            requirements: Algorithm requirements

        Returns:
            Dictionary with recommended node and reasoning
        """
        memory_gb = self.estimate_memory_requirements(requirements)
        llm_heavy = requirements.use_llm_analysis
        ml_training = requirements.use_ml_training

        # Scoring logic
        if ml_training and requirements.use_llm_analysis:
            # Need GPU for local ML training
            recommended = self.nodes[NodeType.RESEARCH]["R4-16-GPU"]
            reason = "GPU required for ML model training in notebooks"
        elif memory_gb > 12 or llm_heavy:
            # Need R8-16
            recommended = self.nodes[NodeType.RESEARCH]["R8-16"]
            reason = f"High memory or LLM ensemble analysis ({memory_gb:.1f}GB)"
        elif memory_gb > 8:
            # Need R4-12
            recommended = self.nodes[NodeType.RESEARCH]["R4-12"]
            reason = f"Moderate research workload ({memory_gb:.1f}GB)"
        elif memory_gb > 4:
            # Can use R2-8
            recommended = self.nodes[NodeType.RESEARCH]["R2-8"]
            reason = f"Light research workload ({memory_gb:.1f}GB)"
        else:
            # Minimal node
            recommended = self.nodes[NodeType.RESEARCH]["R1-4"]
            reason = "Basic research, minimal requirements"

        return {
            "node": recommended,
            "reason": reason,
            "estimated_memory_gb": memory_gb,
            "monthly_cost": recommended.monthly_cost,
        }

    def recommend_live_node(self, requirements: AlgorithmRequirements) -> dict:
        """
        Recommend optimal live trading node.

        Args:
            requirements: Algorithm requirements

        Returns:
            Dictionary with recommended node and reasoning
        """
        memory_gb = self.estimate_memory_requirements(requirements)
        options_intensive = requirements.num_option_chains > 0
        low_latency = requirements.needs_low_latency

        # Scoring logic
        if requirements.use_ml_training:
            # Need GPU for real-time ML
            recommended = self.nodes[NodeType.LIVE_TRADING]["L8-16-GPU"]
            reason = "GPU required for real-time ML inference"
        elif memory_gb > 2 or options_intensive:
            # Need L2-4
            recommended = self.nodes[NodeType.LIVE_TRADING]["L2-4"]
            reason = f"Options trading with multiple chains ({memory_gb:.1f}GB)"
        elif memory_gb > 1:
            # Can use L1-2
            recommended = self.nodes[NodeType.LIVE_TRADING]["L1-2"]
            reason = f"Moderate live trading ({memory_gb:.1f}GB)"
        elif memory_gb > 0.5:
            # Can use L1-1
            recommended = self.nodes[NodeType.LIVE_TRADING]["L1-1"]
            reason = f"Light live trading ({memory_gb:.1f}GB)"
        else:
            # Minimal node (risky for options)
            recommended = self.nodes[NodeType.LIVE_TRADING]["L-MICRO"]
            reason = "CAUTION: Minimal node, not suitable for options"

        return {
            "node": recommended,
            "reason": reason,
            "estimated_memory_gb": memory_gb,
            "monthly_cost": recommended.monthly_cost,
            "colocated": recommended.colocated,
            "max_latency_ms": recommended.max_latency_ms,
        }

    def recommend_nodes(self, requirements: AlgorithmRequirements) -> dict:
        """
        Get recommendations for all node types.

        Args:
            requirements: Algorithm requirements

        Returns:
            Dictionary with recommendations for each node type
        """
        return {
            "backtesting": self.recommend_backtesting_node(requirements),
            "research": self.recommend_research_node(requirements),
            "live_trading": self.recommend_live_node(requirements),
            "total_monthly_cost": sum(
                [
                    self.recommend_backtesting_node(requirements)["monthly_cost"],
                    self.recommend_research_node(requirements)["monthly_cost"],
                    self.recommend_live_node(requirements)["monthly_cost"],
                ]
            ),
        }

    def get_node_comparison(self, node_type: NodeType) -> list[dict]:
        """
        Get comparison table for all nodes of a given type.

        Args:
            node_type: Type of nodes to compare

        Returns:
            List of node specifications
        """
        nodes = []
        for model, spec in self.nodes[node_type].items():
            nodes.append(
                {
                    "model": spec.model,
                    "cores": spec.cores,
                    "speed_ghz": spec.speed_ghz,
                    "ram_gb": spec.ram_gb,
                    "gpu": spec.gpu,
                    "monthly_cost": spec.monthly_cost,
                    "colocated": spec.colocated,
                }
            )
        return nodes


def analyze_algorithm_requirements(
    num_securities: int = 10,
    num_option_chains: int = 5,
    contracts_per_chain: int = 100,
    max_concurrent_positions: int = 20,
    use_llm: bool = True,
    use_ml: bool = False,
) -> dict:
    """
    Convenience function to analyze requirements and get recommendations.

    Args:
        num_securities: Number of equity securities
        num_option_chains: Number of option chains
        contracts_per_chain: Contracts per chain
        max_concurrent_positions: Max concurrent positions
        use_llm: Whether using LLM analysis
        use_ml: Whether using ML training

    Returns:
        Node recommendations dictionary
    """
    optimizer = NodeOptimizer()

    requirements = AlgorithmRequirements(
        num_securities=num_securities,
        num_option_chains=num_option_chains,
        contracts_per_chain=contracts_per_chain,
        max_concurrent_positions=max_concurrent_positions,
        use_llm_analysis=use_llm,
        use_ml_training=use_ml,
    )

    return optimizer.recommend_nodes(requirements)


def print_recommendations(recommendations: dict) -> None:
    """Print recommendations in readable format."""
    print("\n" + "=" * 60)
    print("QUANTCONNECT COMPUTE NODE RECOMMENDATIONS")
    print("=" * 60)

    for category, rec in recommendations.items():
        if category == "total_monthly_cost":
            continue

        print(f"\n{category.upper().replace('_', ' ')}:")
        print(f"  Recommended: {rec['node'].model}")
        print(f"  Cores: {rec['node'].cores} @ {rec['node'].speed_ghz}GHz")
        print(f"  RAM: {rec['node'].ram_gb}GB")
        print(f"  Cost: ${rec['monthly_cost']}/month")
        print(f"  Reason: {rec['reason']}")
        print(f"  Estimated Memory: {rec['estimated_memory_gb']:.2f}GB")

    print(f"\nTOTAL MONTHLY COST: ${recommendations['total_monthly_cost']}/month")
    print("=" * 60 + "\n")
