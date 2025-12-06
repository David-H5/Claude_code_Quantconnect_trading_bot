#!/usr/bin/env python3
"""
Backup Manager for QuantConnect Trading Bot

This module provides automated backup functionality for:
- Algorithm code changes
- Configuration files
- Test results
- Backtest results

Backups are stored locally and can be pushed to a backup branch on GitHub.

Author: QuantConnect Trading Bot
Date: 2025-11-25
"""

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


class BackupManager:
    """
    Manages backups for the trading bot project.

    Provides functionality to:
    - Create timestamped backups of files before changes
    - Restore files from backups
    - Maintain backup history
    - Push backups to GitHub backup branch
    """

    def __init__(
        self,
        project_root: Path | None = None,
        backup_dir: str = ".backups",
        max_backups_per_file: int = 10,
    ):
        """
        Initialize BackupManager.

        Args:
            project_root: Root directory of the project
            backup_dir: Name of backup directory
            max_backups_per_file: Maximum backups to keep per file
        """
        self.project_root = project_root or Path(__file__).parent.parent
        self.backup_dir = self.project_root / backup_dir
        self.max_backups_per_file = max_backups_per_file
        self.manifest_file = self.backup_dir / "manifest.json"

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize manifest
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Load backup manifest from file."""
        if self.manifest_file.exists():
            with open(self.manifest_file) as f:
                return json.load(f)
        return {"backups": {}, "created": datetime.now().isoformat()}

    def _save_manifest(self) -> None:
        """Save backup manifest to file."""
        self._manifest["updated"] = datetime.now().isoformat()
        with open(self.manifest_file, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def _get_file_hash(self, filepath: Path) -> str:
        """Calculate MD5 hash of file contents."""
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_backup_path(self, filepath: Path, timestamp: str) -> Path:
        """Generate backup path for a file."""
        relative_path = filepath.relative_to(self.project_root)
        backup_name = f"{relative_path.stem}_{timestamp}{relative_path.suffix}"
        backup_subdir = self.backup_dir / relative_path.parent
        backup_subdir.mkdir(parents=True, exist_ok=True)
        return backup_subdir / backup_name

    def backup_file(
        self,
        filepath: Path,
        reason: str = "manual backup",
        metadata: dict | None = None,
    ) -> Path | None:
        """
        Create a backup of a file.

        Args:
            filepath: Path to file to backup
            reason: Reason for backup
            metadata: Additional metadata to store

        Returns:
            Path to backup file, or None if backup failed
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self._get_backup_path(filepath, timestamp)

        # Copy file to backup location
        shutil.copy2(filepath, backup_path)

        # Update manifest
        relative_path = str(filepath.relative_to(self.project_root))
        if relative_path not in self._manifest["backups"]:
            self._manifest["backups"][relative_path] = []

        backup_info = {
            "timestamp": timestamp,
            "backup_path": str(backup_path.relative_to(self.project_root)),
            "original_hash": self._get_file_hash(filepath),
            "reason": reason,
            "metadata": metadata or {},
        }

        self._manifest["backups"][relative_path].insert(0, backup_info)

        # Prune old backups
        self._prune_backups(relative_path)

        self._save_manifest()
        return backup_path

    def _prune_backups(self, relative_path: str) -> None:
        """Remove old backups beyond the maximum limit."""
        backups = self._manifest["backups"].get(relative_path, [])
        if len(backups) > self.max_backups_per_file:
            # Remove oldest backups
            for old_backup in backups[self.max_backups_per_file :]:
                old_path = self.project_root / old_backup["backup_path"]
                if old_path.exists():
                    old_path.unlink()
            self._manifest["backups"][relative_path] = backups[: self.max_backups_per_file]

    def restore_file(
        self,
        filepath: Path,
        backup_index: int = 0,
    ) -> bool:
        """
        Restore a file from backup.

        Args:
            filepath: Path to file to restore
            backup_index: Index of backup to restore (0 = most recent)

        Returns:
            True if restore successful, False otherwise
        """
        filepath = Path(filepath)
        relative_path = str(filepath.relative_to(self.project_root))

        backups = self._manifest["backups"].get(relative_path, [])
        if not backups or backup_index >= len(backups):
            return False

        backup_info = backups[backup_index]
        backup_path = self.project_root / backup_info["backup_path"]

        if not backup_path.exists():
            return False

        # Create backup of current file before restoring
        if filepath.exists():
            self.backup_file(filepath, reason="pre-restore backup")

        # Restore the file
        shutil.copy2(backup_path, filepath)
        return True

    def list_backups(self, filepath: Path | None = None) -> dict[str, list[dict]]:
        """
        List available backups.

        Args:
            filepath: Specific file to list backups for, or None for all

        Returns:
            Dictionary of file paths to backup info lists
        """
        if filepath:
            relative_path = str(filepath.relative_to(self.project_root))
            return {relative_path: self._manifest["backups"].get(relative_path, [])}
        return self._manifest["backups"]

    def backup_algorithms(self, reason: str = "algorithm backup") -> list[Path]:
        """Backup all algorithm files."""
        algorithms_dir = self.project_root / "algorithms"
        backed_up = []

        for algo_file in algorithms_dir.glob("*.py"):
            if algo_file.name != "__init__.py":
                backup_path = self.backup_file(algo_file, reason=reason)
                if backup_path:
                    backed_up.append(backup_path)

        return backed_up

    def backup_before_change(
        self,
        filepath: Path,
        change_description: str,
    ) -> Path | None:
        """
        Create a backup before making changes to a file.

        This is the recommended method to call before any file modification.

        Args:
            filepath: File that will be changed
            change_description: Description of the planned change

        Returns:
            Path to backup file
        """
        return self.backup_file(
            filepath,
            reason=f"pre-change: {change_description}",
            metadata={"change_type": "pre_modification"},
        )

    def create_checkpoint(self, name: str, files: list[Path]) -> str:
        """
        Create a named checkpoint of multiple files.

        Args:
            name: Name for this checkpoint
            files: List of files to include

        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for filepath in files:
            self.backup_file(
                filepath,
                reason=f"checkpoint: {name}",
                metadata={"checkpoint_id": checkpoint_id},
            )

        return checkpoint_id

    def push_to_backup_branch(self) -> tuple[bool, str]:
        """
        Push backups to a dedicated backup branch on GitHub.

        Returns:
            Tuple of (success, message)
        """
        try:
            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            current_branch = result.stdout.strip()

            # Create or switch to backup branch
            backup_branch = "backups/automated"
            subprocess.run(
                ["git", "checkout", "-B", backup_branch],
                capture_output=True,
                cwd=self.project_root,
            )

            # Add and commit backup files
            subprocess.run(
                ["git", "add", str(self.backup_dir)],
                capture_output=True,
                cwd=self.project_root,
            )

            commit_msg = f"Automated backup: {datetime.now().isoformat()}"
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True,
                cwd=self.project_root,
            )

            # Push to remote
            subprocess.run(
                ["git", "push", "-u", "origin", backup_branch, "--force"],
                capture_output=True,
                cwd=self.project_root,
            )

            # Switch back to original branch
            subprocess.run(
                ["git", "checkout", current_branch],
                capture_output=True,
                cwd=self.project_root,
            )

            return True, f"Backups pushed to {backup_branch}"

        except Exception as e:
            return False, str(e)


def create_pre_change_backup(filepath: str, description: str) -> str | None:
    """
    Convenience function to create a backup before changes.

    Args:
        filepath: Path to file
        description: Description of planned change

    Returns:
        Path to backup file as string, or None
    """
    manager = BackupManager()
    backup_path = manager.backup_before_change(Path(filepath), description)
    return str(backup_path) if backup_path else None


if __name__ == "__main__":
    # Example usage
    manager = BackupManager()

    # Backup all algorithms
    backed_up = manager.backup_algorithms("manual backup before testing")
    print(f"Backed up {len(backed_up)} algorithm files")

    # List all backups
    all_backups = manager.list_backups()
    for filepath, backups in all_backups.items():
        print(f"\n{filepath}: {len(backups)} backups")
        for b in backups[:3]:
            print(f"  - {b['timestamp']}: {b['reason']}")
