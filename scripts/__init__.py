"""
Scripts and Utilities for QuantConnect Trading Bot

This package contains utility scripts for:
- Backup management
- Algorithm validation
- Deployment safety checks
- Autonomous testing
"""

from .algorithm_validator import AlgorithmValidator, validate_algorithm, validate_all_algorithms
from .backup_manager import BackupManager, create_pre_change_backup


__all__ = [
    "AlgorithmValidator",
    "BackupManager",
    "create_pre_change_backup",
    "validate_algorithm",
    "validate_all_algorithms",
]
