#!/usr/bin/env python3
"""
Deploy Trading Algorithm to QuantConnect with Optimal Node Selection

Automatically selects appropriate compute nodes (B8-16, R8-16, L2-4)
based on algorithm requirements and deploys to QuantConnect.

Author: QuantConnect Trading Bot
Date: 2025-11-30
"""

import argparse
import logging
import os
import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from utils.node_optimizer import (
    AlgorithmRequirements,
    NodeOptimizer,
    print_recommendations,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_algorithm_file(algorithm_path: Path) -> AlgorithmRequirements:
    """
    Analyze algorithm file to estimate resource requirements.

    Args:
        algorithm_path: Path to algorithm file

    Returns:
        Estimated algorithm requirements
    """
    logger.info(f"Analyzing algorithm: {algorithm_path}")

    # Default requirements (can be overridden)
    requirements = AlgorithmRequirements(
        num_securities=10,
        num_option_chains=5,
        contracts_per_chain=100,
        max_concurrent_positions=20,
        use_llm_analysis=False,
        use_ml_training=False,
        tick_data=False,
        long_backtest_days=365,
        needs_low_latency=True,
    )

    # Read algorithm file
    with open(algorithm_path) as f:
        content = f.read()

    # Heuristics to detect features
    if "AddOption" in content:
        requirements.num_option_chains = 5
        logger.info("Detected options trading")

    if any(x in content for x in ["ensemble", "llm", "sentiment"]):
        requirements.use_llm_analysis = True
        logger.info("Detected LLM analysis")

    if any(x in content for x in ["tensorflow", "torch", "sklearn", "train", "fit"]):
        requirements.use_ml_training = True
        logger.info("Detected ML training")

    if "Resolution.Tick" in content:
        requirements.tick_data = True
        logger.info("Detected tick data usage")

    # Count universe size heuristic
    add_equity_count = content.count("AddEquity")
    if add_equity_count > 0:
        requirements.num_securities = add_equity_count
        logger.info(f"Estimated {add_equity_count} securities")

    return requirements


def deploy_to_quantconnect(
    algorithm_path: Path,
    node_type: str,
    node_model: str,
    project_id: str | None = None,
    dry_run: bool = False,
) -> bool:
    """
    Deploy algorithm to QuantConnect with specified node.

    Args:
        algorithm_path: Path to algorithm file
        node_type: Type of deployment (backtest, research, live)
        node_model: Node model to use (e.g., "B8-16")
        project_id: QuantConnect project ID
        dry_run: If True, don't actually deploy

    Returns:
        True if deployment successful
    """
    logger.info(f"Deploying to QuantConnect: {node_type} on {node_model}")

    if dry_run:
        logger.info("DRY RUN: Would deploy with LEAN CLI")
        logger.info(f"  Algorithm: {algorithm_path}")
        logger.info(f"  Node: {node_model}")
        logger.info(f"  Type: {node_type}")
        return True

    # Load config
    config = get_config()
    qc_config = config.get("quantconnect", {})

    user_id = qc_config.get("user_id") or os.getenv("QC_USER_ID")
    api_token = qc_config.get("api_token") or os.getenv("QC_API_TOKEN")

    if not user_id or not api_token:
        logger.error("Missing QuantConnect credentials")
        logger.error("Set QC_USER_ID and QC_API_TOKEN environment variables")
        return False

    # Determine LEAN CLI command
    if node_type == "backtest":
        # Deploy backtest
        cmd = f"lean cloud backtest '{algorithm_path.parent}' --node {node_model}"
    elif node_type == "research":
        # Launch research notebook
        cmd = f"lean cloud research --node {node_model}"
    elif node_type == "live":
        # Deploy live trading
        if not project_id:
            logger.error("Project ID required for live deployment")
            return False
        cmd = f"lean cloud live '{algorithm_path.parent}' --node {node_model} --brokerage schwab"
    else:
        logger.error(f"Unknown node type: {node_type}")
        return False

    logger.info(f"Executing: {cmd}")

    # Execute LEAN CLI command
    import subprocess

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Deployment successful!")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Deployment failed: {e}")
        logger.error(e.stderr)
        return False


def main():
    """Main deployment workflow."""
    parser = argparse.ArgumentParser(description="Deploy algorithm to QuantConnect with optimal node selection")
    parser.add_argument(
        "algorithm",
        type=Path,
        help="Path to algorithm file",
    )
    parser.add_argument(
        "--type",
        choices=["backtest", "research", "live"],
        default="backtest",
        help="Deployment type (default: backtest)",
    )
    parser.add_argument(
        "--node",
        help="Override node selection (e.g., B8-16, R8-16, L2-4)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze requirements, don't deploy",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deployed without deploying",
    )
    parser.add_argument(
        "--project-id",
        help="QuantConnect project ID (for live deployment)",
    )

    args = parser.parse_args()

    # Validate algorithm file
    if not args.algorithm.exists():
        logger.error(f"Algorithm file not found: {args.algorithm}")
        sys.exit(1)

    # Analyze algorithm requirements
    logger.info("=" * 60)
    logger.info("ANALYZING ALGORITHM REQUIREMENTS")
    logger.info("=" * 60)

    requirements = analyze_algorithm_file(args.algorithm)

    # Get node recommendations
    optimizer = NodeOptimizer()
    recommendations = optimizer.recommend_nodes(requirements)

    # Print recommendations
    print_recommendations(recommendations)

    # If analyze-only, exit here
    if args.analyze_only:
        logger.info("Analysis complete (--analyze-only specified)")
        sys.exit(0)

    # Determine which node to use
    if args.node:
        # User override
        node_model = args.node
        logger.info(f"Using user-specified node: {node_model}")
    else:
        # Use recommended node
        if args.type == "backtest":
            node_model = recommendations["backtesting"]["node"].model
        elif args.type == "research":
            node_model = recommendations["research"]["node"].model
        elif args.type == "live":
            node_model = recommendations["live_trading"]["node"].model

        logger.info(f"Using recommended node: {node_model}")

    # Confirm deployment
    if not args.dry_run:
        print("\n" + "=" * 60)
        print("DEPLOYMENT CONFIRMATION")
        print("=" * 60)
        print(f"Algorithm: {args.algorithm}")
        print(f"Type: {args.type}")
        print(f"Node: {node_model}")
        print("=" * 60)

        response = input("\nProceed with deployment? [y/N]: ")
        if response.lower() not in ["y", "yes"]:
            logger.info("Deployment cancelled by user")
            sys.exit(0)

    # Deploy
    success = deploy_to_quantconnect(
        algorithm_path=args.algorithm,
        node_type=args.type,
        node_model=node_model,
        project_id=args.project_id,
        dry_run=args.dry_run,
    )

    if success:
        logger.info("✓ Deployment completed successfully")
        sys.exit(0)
    else:
        logger.error("✗ Deployment failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
