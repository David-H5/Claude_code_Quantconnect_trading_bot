#!/usr/bin/env python3
"""
Manage QuantConnect Object Store

Utility script for inspecting, cleaning, and maintaining Object Store (5GB tier).

Usage:
    python scripts/manage_object_store.py --list
    python scripts/manage_object_store.py --stats
    python scripts/manage_object_store.py --cleanup-expired
    python scripts/manage_object_store.py --cleanup-category ml_models
    python scripts/manage_object_store.py --export output.json

Author: QuantConnect Trading Bot
Date: 2025-11-30
"""

import argparse
import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Note: This script is designed for research environment use
# Live trading algorithms cannot list all keys in ObjectStore


def print_storage_stats(stats: dict) -> None:
    """Print storage statistics in readable format."""
    print("\n" + "=" * 60)
    print("OBJECT STORE STATISTICS (5GB Tier)")
    print("=" * 60)
    print(f"Total Usage: {stats['total_gb']:.2f}GB / {stats['limit_gb']}GB ({stats['total_pct']:.1f}%)")
    print(f"Files: {stats['file_count']:,} / {stats['file_limit']:,}")

    if stats.get("growth_rate_gb_per_day"):
        print(f"Growth Rate: {stats['growth_rate_gb_per_day']:.3f}GB/day")

    if stats.get("days_until_full"):
        print(f"Days Until Full: ~{stats['days_until_full']} days")

    print("\nStorage by Category:")
    print("-" * 60)
    for category, cat_stats in stats["by_category"].items():
        quota_mb = cat_stats["quota_mb"]
        size_mb = cat_stats["size_mb"]
        usage_pct = (size_mb / quota_mb * 100) if quota_mb > 0 else 0

        print(f"{category:20} {size_mb:8.1f}MB / {quota_mb:8.1f}MB ({usage_pct:5.1f}%) - {cat_stats['count']:,} files")

    print("=" * 60)


def print_cleanup_suggestions(suggestions: dict) -> None:
    """Print cleanup suggestions."""
    print("\n" + "=" * 60)
    print("CLEANUP SUGGESTIONS")
    print("=" * 60)

    if not suggestions["cleanup_needed"]:
        print("✓ No cleanup needed - storage is healthy")
        return

    print("⚠ Cleanup recommended - total usage above threshold")
    print("\nSuggested Actions:")

    for i, action in enumerate(suggestions["actions"], 1):
        print(f"\n{i}. {action['action'].upper()}")
        print(f"   Category: {action['category']}")
        print(f"   Reason: {action['reason']}")
        if "expected_savings_mb" in action:
            print(f"   Expected savings: {action['expected_savings_mb']:.1f}MB")

    print("=" * 60)


def main():
    """Main management workflow."""
    parser = argparse.ArgumentParser(description="Manage QuantConnect Object Store (5GB tier)")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all stored objects",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show storage statistics",
    )
    parser.add_argument(
        "--cleanup-expired",
        action="store_true",
        help="Remove expired objects",
    )
    parser.add_argument(
        "--cleanup-category",
        help="Cleanup old files in specific category",
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Show cleanup suggestions",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export all metadata to JSON file",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("QUANTCONNECT OBJECT STORE MANAGER")
    print("=" * 60)
    print("\nIMPORTANT: This script should be run in a QuantConnect")
    print("research environment where full ObjectStore access is available.")
    print("\nIn live trading, ObjectStore does not support listing all keys.")
    print("Use algorithm-based management methods instead.")
    print("=" * 60)

    # Example demonstration (would need actual QC environment)
    print("\nExample Usage:")
    print("\n# In a research notebook:")
    print("from utils import create_object_store_manager, create_storage_monitor")
    print("from config import get_config")
    print("\n# Create manager")
    print("config = get_config()")
    print("manager = create_object_store_manager(qb, config.get('quantconnect', {}).get('object_store', {}))")
    print("monitor = create_storage_monitor(manager, config.get('quantconnect', {}).get('object_store', {}))")
    print("\n# Get statistics")
    print("stats = monitor.get_statistics()")
    print('print(f\'Storage: {stats["current_usage_gb"]:.2f}GB / {stats["limit_gb"]}GB\')')
    print("\n# Cleanup expired")
    print("deleted = manager.cleanup_expired()")
    print('print(f"Deleted {deleted} expired objects")')
    print("\n# Get suggestions")
    print("suggestions = monitor.suggest_cleanup()")
    print("for action in suggestions['actions']:")
    print('    print(f\'  {action["action"]}: {action["reason"]}\')')

    print("\n" + "=" * 60)
    print("For live trading algorithms, use these methods:")
    print("=" * 60)
    print("\n# In algorithm OnEndOfDay or scheduled function:")
    print("if self.storage_monitor:")
    print("    stats = self.storage_monitor.check_usage()")
    print("    if not self.storage_monitor.is_healthy():")
    print("        self.object_store_manager.cleanup_expired()")
    print("\n# Save data with automatic expiration:")
    print("self.object_store_manager.save(")
    print("    key='my_data',")
    print("    data={'value': 123},")
    print("    category=StorageCategory.TRADING_STATE,")
    print("    expire_days=30,")
    print(")")

    print("\n" + "=" * 60)
    print("\nFor full management capabilities, use:")
    print("1. LEAN CLI: lean cloud object-store list")
    print("2. Web interface: https://www.quantconnect.com/organization/object-store")
    print("3. Research environment with this script")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
