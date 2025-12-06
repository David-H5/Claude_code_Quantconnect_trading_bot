#!/usr/bin/env python3
"""
Claude Code Registry Sync Script

Automatically discovers, registers, and configures Claude Code hooks, scripts,
commands, and workflows. Ensures mandatory components are properly configured.

Usage:
    python scripts/sync_claude_registry.py              # Full sync
    python scripts/sync_claude_registry.py --check      # Check only, no changes
    python scripts/sync_claude_registry.py --discover   # Discover new components
    python scripts/sync_claude_registry.py --enforce    # Enforce mandatory hooks
    python scripts/sync_claude_registry.py --validate   # Validate all components

Features:
- Auto-discovers hooks, scripts, commands in .claude/ and scripts/
- Updates registry.json with new components
- Syncs settings.json with registry (ensures mandatory hooks configured)
- Validates all components work correctly
- Cross-references documentation
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SyncResult:
    """Result of a sync operation."""

    action: str
    component: str
    status: str  # "ok", "added", "fixed", "error", "warning"
    message: str


class ClaudeRegistrySync:
    """Manages Claude Code registry synchronization."""

    def __init__(self, project_root: Path, check_only: bool = False):
        self.project_root = project_root
        self.claude_dir = project_root / ".claude"
        self.scripts_dir = project_root / "scripts"
        self.check_only = check_only
        self.results: list[SyncResult] = []

        # Load current state
        self.registry = self._load_json(self.claude_dir / "registry.json")
        self.settings = self._load_json(self.claude_dir / "settings.json")

    def _load_json(self, path: Path) -> dict:
        """Load JSON file or return empty dict."""
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                self.results.append(SyncResult("load", str(path.name), "error", f"Invalid JSON: {e}"))
        return {}

    def _save_json(self, path: Path, data: dict) -> bool:
        """Save JSON file with pretty formatting."""
        if self.check_only:
            return True
        try:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return True
        except OSError as e:
            self.results.append(SyncResult("save", str(path.name), "error", f"Write failed: {e}"))
            return False

    def discover_hooks(self) -> dict[str, dict]:
        """Discover all hooks in .claude/hooks/."""
        discovered = {}
        hooks_dir = self.claude_dir / "hooks"

        if not hooks_dir.exists():
            return discovered

        for py_file in hooks_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            hook_name = py_file.stem
            content = py_file.read_text(encoding="utf-8")

            # Extract metadata from docstring
            description = self._extract_docstring_summary(content)
            trigger = self._detect_hook_trigger(content)
            matcher = self._detect_hook_matcher(content)

            discovered[hook_name] = {
                "file": f"hooks/{py_file.name}",
                "trigger": trigger,
                "matcher": matcher,
                "mandatory": False,  # Default, can be overridden
                "description": description,
                "documentation": None,
            }

        return discovered

    def discover_scripts(self) -> dict[str, dict]:
        """Discover all scripts in scripts/."""
        discovered = {}

        if not self.scripts_dir.exists():
            return discovered

        for py_file in self.scripts_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            script_name = py_file.stem
            content = py_file.read_text(encoding="utf-8")

            # Extract metadata
            description = self._extract_docstring_summary(content)
            usage = self._extract_usage(content)
            category = self._detect_script_category(script_name, content)

            discovered[script_name] = {
                "file": f"../scripts/{py_file.name}",
                "category": category,
                "mandatory": False,
                "description": description,
                "usage": usage,
                "documentation": None,
            }

        return discovered

    def discover_commands(self) -> dict[str, dict]:
        """Discover all commands in .claude/commands/."""
        discovered = {}
        commands_dir = self.claude_dir / "commands"

        if not commands_dir.exists():
            return discovered

        for md_file in commands_dir.glob("*.md"):
            command_name = md_file.stem
            content = md_file.read_text(encoding="utf-8")

            # Extract first heading as description
            description = self._extract_first_heading(content)
            category = self._detect_command_category(command_name, content)

            discovered[command_name] = {
                "file": f"commands/{md_file.name}",
                "category": category,
                "mandatory": False,
                "description": description,
                "documentation": None,
            }

        return discovered

    def _extract_docstring_summary(self, content: str) -> str:
        """Extract summary from Python docstring."""
        match = re.search(r'"""(.+?)"""', content, re.DOTALL)
        if match:
            lines = match.group(1).strip().split("\n")
            # Return first non-empty line
            for line in lines:
                line = line.strip()
                if line and not line.startswith("Usage"):
                    return line[:100]
        return "No description"

    def _extract_usage(self, content: str) -> str | None:
        """Extract usage example from docstring."""
        match = re.search(r"Usage:\s*\n\s*(.+?)(?:\n\s*\n|Options:|$)", content, re.DOTALL)
        if match:
            usage = match.group(1).strip().split("\n")[0].strip()
            return usage
        return None

    def _extract_first_heading(self, content: str) -> str:
        """Extract first markdown heading."""
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "No description"

    def _detect_hook_trigger(self, content: str) -> str:
        """Detect hook trigger from content."""
        if "PreToolUse" in content:
            return "PreToolUse"
        elif "PostToolUse" in content:
            return "PostToolUse"
        elif "SessionStart" in content:
            return "SessionStart"
        elif "UserPromptSubmit" in content:
            return "UserPromptSubmit"
        return "PostToolUse"  # Default

    def _detect_hook_matcher(self, content: str) -> str | None:
        """Detect hook matcher pattern from content."""
        # Look for tool_name checks
        if "WebSearch" in content and "WebFetch" in content:
            return "WebSearch|WebFetch"
        elif "Edit" in content and "Write" in content:
            if "Read" in content:
                return "Edit|Write|Read"
            return "Edit|Write"
        return None

    def _detect_script_category(self, name: str, content: str) -> str:
        """Detect script category."""
        if "research" in name or "doc" in name:
            return "documentation"
        elif "valid" in name:
            return "validation"
        elif "backup" in name or "watchdog" in name:
            return "safety"
        elif "deploy" in name:
            return "deployment"
        elif "analyze" in name:
            return "analysis"
        return "general"

    def _detect_command_category(self, name: str, content: str) -> str:
        """Detect command category."""
        if "ric" in name:
            return "workflow"
        elif "doc" in name or "valid" in name:
            return "documentation"
        return "development"

    def sync_registry(self) -> None:
        """Sync discovered components with registry."""
        # Discover all components
        discovered_hooks = self.discover_hooks()
        discovered_scripts = self.discover_scripts()
        discovered_commands = self.discover_commands()

        # Merge with existing registry
        existing_hooks = self.registry.get("hooks", {})
        existing_scripts = self.registry.get("scripts", {})
        existing_commands = self.registry.get("commands", {})

        # Add new hooks
        for name, info in discovered_hooks.items():
            if name not in existing_hooks:
                existing_hooks[name] = info
                self.results.append(SyncResult("discover", f"hook:{name}", "added", "New hook discovered"))
            else:
                # Update file path if changed
                if existing_hooks[name].get("file") != info["file"]:
                    existing_hooks[name]["file"] = info["file"]

        # Add new scripts
        for name, info in discovered_scripts.items():
            if name not in existing_scripts:
                existing_scripts[name] = info
                self.results.append(SyncResult("discover", f"script:{name}", "added", "New script discovered"))

        # Add new commands
        for name, info in discovered_commands.items():
            if name not in existing_commands:
                existing_commands[name] = info
                self.results.append(SyncResult("discover", f"command:{name}", "added", "New command discovered"))

        # Update registry
        self.registry["hooks"] = existing_hooks
        self.registry["scripts"] = existing_scripts
        self.registry["commands"] = existing_commands
        self.registry["updated"] = datetime.now().strftime("%Y-%m-%d")

        if not self.check_only:
            self._save_json(self.claude_dir / "registry.json", self.registry)

    def sync_settings(self) -> None:
        """Sync settings.json with registry to ensure hooks are configured."""
        if not self.settings.get("hooks"):
            self.settings["hooks"] = {}

        hooks_config = self.settings["hooks"]
        registry_hooks = self.registry.get("hooks", {})

        # Group hooks by trigger and matcher
        hook_groups: dict[str, dict[str, list[dict]]] = {}

        for hook_name, hook_info in registry_hooks.items():
            trigger = hook_info.get("trigger", "PostToolUse")
            matcher = hook_info.get("matcher")

            if trigger not in hook_groups:
                hook_groups[trigger] = {}

            matcher_key = matcher or "__none__"
            if matcher_key not in hook_groups[trigger]:
                hook_groups[trigger][matcher_key] = []

            hook_groups[trigger][matcher_key].append({"name": hook_name, "info": hook_info})

        # Check each trigger type
        for trigger, matchers in hook_groups.items():
            if trigger not in hooks_config:
                hooks_config[trigger] = []

            existing_configs = hooks_config[trigger]

            for matcher_key, hooks in matchers.items():
                matcher = None if matcher_key == "__none__" else matcher_key

                # Find existing config for this matcher
                existing_config = None
                for cfg in existing_configs:
                    cfg_matcher = cfg.get("matcher")
                    if cfg_matcher == matcher:
                        existing_config = cfg
                        break

                if existing_config is None:
                    # Create new config
                    existing_config = {"hooks": []}
                    if matcher:
                        existing_config["matcher"] = matcher
                    existing_configs.append(existing_config)

                # Check each hook is configured
                existing_commands = {h.get("command", ""): h for h in existing_config.get("hooks", [])}

                for hook in hooks:
                    hook_name = hook["name"]
                    hook_info = hook["info"]
                    hook_file = hook_info.get("file", "")

                    # Build expected command
                    expected_cmd = f"cd {self.project_root} && " f"python3 .claude/{hook_file}"

                    # Check if already configured
                    is_configured = any(hook_file in cmd for cmd in existing_commands)

                    if not is_configured:
                        # Add hook configuration
                        hook_config = {"type": "command", "command": expected_cmd}

                        # Add status message for visibility
                        if hook_info.get("description"):
                            desc = hook_info["description"][:50]
                            hook_config["statusMessage"] = desc

                        existing_config["hooks"].append(hook_config)

                        status = "added" if not self.check_only else "warning"
                        self.results.append(
                            SyncResult("sync", f"settings:{hook_name}", status, "Hook added to settings.json")
                        )

        self.settings["hooks"] = hooks_config

        if not self.check_only:
            self._save_json(self.claude_dir / "settings.json", self.settings)

    def enforce_mandatory(self) -> None:
        """Ensure all mandatory hooks are configured."""
        mandatory_hooks = self.registry.get("mandatory_hooks", [])
        registry_hooks = self.registry.get("hooks", {})

        for hook_name in mandatory_hooks:
            if hook_name not in registry_hooks:
                self.results.append(
                    SyncResult("enforce", f"hook:{hook_name}", "error", "Mandatory hook not in registry")
                )
                continue

            hook_info = registry_hooks[hook_name]
            hook_file = hook_info.get("file", "")
            full_path = self.claude_dir / hook_file

            # Check file exists
            if not full_path.exists():
                self.results.append(
                    SyncResult("enforce", f"hook:{hook_name}", "error", f"Hook file not found: {hook_file}")
                )
                continue

            # Check configured in settings
            is_configured = False
            for trigger_hooks in self.settings.get("hooks", {}).values():
                for cfg in trigger_hooks:
                    for h in cfg.get("hooks", []):
                        if hook_file in h.get("command", ""):
                            is_configured = True
                            break

            if not is_configured:
                self.results.append(
                    SyncResult("enforce", f"hook:{hook_name}", "error", "Mandatory hook not in settings.json")
                )
            else:
                self.results.append(SyncResult("enforce", f"hook:{hook_name}", "ok", "Mandatory hook configured"))

    def validate_components(self) -> None:
        """Validate all components work correctly."""
        registry_hooks = self.registry.get("hooks", {})

        for hook_name, hook_info in registry_hooks.items():
            hook_file = hook_info.get("file", "")
            full_path = self.claude_dir / hook_file

            # Check file exists
            if not full_path.exists():
                self.results.append(SyncResult("validate", f"hook:{hook_name}", "error", "File not found"))
                continue

            # Check syntax
            try:
                result = subprocess.run(
                    ["python3", "-m", "py_compile", str(full_path)], capture_output=True, text=True, timeout=5
                )
                if result.returncode != 0:
                    self.results.append(
                        SyncResult("validate", f"hook:{hook_name}", "error", f"Syntax error: {result.stderr[:100]}")
                    )
                else:
                    self.results.append(SyncResult("validate", f"hook:{hook_name}", "ok", "Syntax valid"))
            except subprocess.TimeoutExpired:
                self.results.append(SyncResult("validate", f"hook:{hook_name}", "warning", "Validation timeout"))
            except Exception as e:
                self.results.append(SyncResult("validate", f"hook:{hook_name}", "error", f"Validation failed: {e}"))

        # Validate scripts
        registry_scripts = self.registry.get("scripts", {})
        for script_name, script_info in registry_scripts.items():
            script_file = script_info.get("file", "")
            # Handle relative path
            if script_file.startswith("../"):
                full_path = self.claude_dir / script_file
            else:
                full_path = self.project_root / script_file

            full_path = full_path.resolve()

            if not full_path.exists():
                self.results.append(
                    SyncResult("validate", f"script:{script_name}", "error", f"File not found: {full_path}")
                )
            else:
                self.results.append(SyncResult("validate", f"script:{script_name}", "ok", "File exists"))

    def check_cross_references(self) -> None:
        """Check documentation cross-references."""
        registry_hooks = self.registry.get("hooks", {})

        for hook_name, hook_info in registry_hooks.items():
            doc_path = hook_info.get("documentation")
            if doc_path:
                full_path = (self.claude_dir / doc_path).resolve()
                if not full_path.exists():
                    self.results.append(
                        SyncResult("xref", f"hook:{hook_name}", "warning", f"Documentation not found: {doc_path}")
                    )

    def get_summary(self) -> dict:
        """Get sync summary."""
        by_status = {}
        for r in self.results:
            by_status.setdefault(r.status, []).append(r)

        return {
            "total": len(self.results),
            "ok": len(by_status.get("ok", [])),
            "added": len(by_status.get("added", [])),
            "fixed": len(by_status.get("fixed", [])),
            "warnings": len(by_status.get("warning", [])),
            "errors": len(by_status.get("error", [])),
            "all_ok": len(by_status.get("error", [])) == 0,
        }

    def run_full_sync(self) -> None:
        """Run full synchronization."""
        print("=" * 60)
        print("Claude Code Registry Sync")
        print("=" * 60)

        print("\n1. Discovering components...")
        self.sync_registry()

        print("2. Syncing settings.json...")
        self.sync_settings()

        print("3. Enforcing mandatory hooks...")
        self.enforce_mandatory()

        print("4. Validating components...")
        self.validate_components()

        print("5. Checking cross-references...")
        self.check_cross_references()

        # Print results
        summary = self.get_summary()
        print("\n" + "-" * 60)
        print("Results:")
        print("-" * 60)

        for r in self.results:
            if r.status == "ok":
                icon = "✓"
            elif r.status == "added":
                icon = "+"
            elif r.status == "fixed":
                icon = "~"
            elif r.status == "warning":
                icon = "⚠"
            else:
                icon = "✗"

            print(f"  {icon} [{r.action}] {r.component}: {r.message}")

        print("\n" + "=" * 60)
        print(
            f"Summary: {summary['ok']} ok, {summary['added']} added, "
            f"{summary['warnings']} warnings, {summary['errors']} errors"
        )

        if self.check_only:
            print("(Check only mode - no changes made)")

        if summary["all_ok"]:
            print("✅ All checks passed!")
        else:
            print("❌ Issues found - review above")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sync Claude Code registry and settings")
    parser.add_argument("--check", action="store_true", help="Check only, don't modify files")
    parser.add_argument("--discover", action="store_true", help="Only discover new components")
    parser.add_argument("--enforce", action="store_true", help="Only enforce mandatory hooks")
    parser.add_argument("--validate", action="store_true", help="Only validate components")

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    sync = ClaudeRegistrySync(project_root, check_only=args.check)

    if args.discover:
        sync.sync_registry()
    elif args.enforce:
        sync.enforce_mandatory()
    elif args.validate:
        sync.validate_components()
    else:
        sync.run_full_sync()

    summary = sync.get_summary()
    sys.exit(0 if summary["all_ok"] else 1)


if __name__ == "__main__":
    main()
