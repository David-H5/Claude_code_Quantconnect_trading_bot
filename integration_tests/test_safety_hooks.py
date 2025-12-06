"""
Safety Hooks Integration Tests

UPGRADE-015 Phase 12: Integration & Final Validation

Tests hook system integration:
- Pre-tool use validation
- Post-tool use logging
- Session hooks
- Algorithm change guards
"""

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRiskValidatorHook:
    """Integration tests for risk validator hook."""

    def test_risk_validator_exists(self):
        """Test that risk validator hook file exists."""
        hook_path = Path(".claude/hooks/trading/risk_validator.py")
        assert hook_path.exists(), "risk_validator.py hook not found"

    def test_risk_validator_syntax(self):
        """Test risk validator hook has valid Python syntax."""
        hook_path = Path(".claude/hooks/trading/risk_validator.py")
        code = hook_path.read_text()

        # Should compile without syntax errors
        compile(code, str(hook_path), "exec")

    def test_risk_validator_has_main(self):
        """Test risk validator hook has main function or entry point."""
        hook_path = Path(".claude/hooks/trading/risk_validator.py")
        code = hook_path.read_text()

        # Should have entry point
        assert "def main" in code or "if __name__" in code


class TestLogTradeHook:
    """Integration tests for log trade hook."""

    def test_log_trade_exists(self):
        """Test that log trade hook file exists."""
        hook_path = Path(".claude/hooks/trading/log_trade.py")
        assert hook_path.exists(), "log_trade.py hook not found"

    def test_log_trade_syntax(self):
        """Test log trade hook has valid Python syntax."""
        hook_path = Path(".claude/hooks/trading/log_trade.py")
        code = hook_path.read_text()

        compile(code, str(hook_path), "exec")


class TestLoadContextHook:
    """Integration tests for load context hook."""

    def test_load_context_exists(self):
        """Test that load context hook file exists."""
        hook_path = Path(".claude/hooks/trading/load_trading_context.py")
        assert hook_path.exists(), "load_trading_context.py hook not found"

    def test_load_context_syntax(self):
        """Test load context hook has valid Python syntax."""
        hook_path = Path(".claude/hooks/trading/load_trading_context.py")
        code = hook_path.read_text()

        compile(code, str(hook_path), "exec")


class TestParseBacktestHook:
    """Integration tests for parse backtest hook."""

    def test_parse_backtest_exists(self):
        """Test that parse backtest hook file exists."""
        hook_path = Path(".claude/hooks/trading/parse_backtest.py")
        assert hook_path.exists(), "parse_backtest.py hook not found"

    def test_parse_backtest_syntax(self):
        """Test parse backtest hook has valid Python syntax."""
        hook_path = Path(".claude/hooks/trading/parse_backtest.py")
        code = hook_path.read_text()

        compile(code, str(hook_path), "exec")


class TestAlgoChangeGuardHook:
    """Integration tests for algo change guard hook."""

    def test_algo_change_guard_exists(self):
        """Test that algo change guard hook file exists."""
        hook_path = Path(".claude/hooks/validation/algo_change_guard.py")
        assert hook_path.exists(), "algo_change_guard.py hook not found"

    def test_algo_change_guard_syntax(self):
        """Test algo change guard hook has valid Python syntax."""
        hook_path = Path(".claude/hooks/validation/algo_change_guard.py")
        code = hook_path.read_text()

        compile(code, str(hook_path), "exec")


class TestHooksConfiguration:
    """Integration tests for hooks configuration."""

    def test_settings_exists(self):
        """Test that settings.json exists."""
        settings_path = Path(".claude/settings.json")
        assert settings_path.exists(), "settings.json not found"

    def test_settings_valid_json(self):
        """Test that settings.json is valid JSON."""
        settings_path = Path(".claude/settings.json")
        content = settings_path.read_text()

        # Should parse without error
        settings = json.loads(content)
        assert isinstance(settings, dict)

    def test_hooks_section_exists(self):
        """Test that hooks section exists in settings."""
        settings_path = Path(".claude/settings.json")
        content = settings_path.read_text()
        settings = json.loads(content)

        assert "hooks" in settings, "hooks section not in settings.json"

    def test_pretooluse_hooks_configured(self):
        """Test that PreToolUse hooks are configured."""
        settings_path = Path(".claude/settings.json")
        content = settings_path.read_text()
        settings = json.loads(content)

        hooks = settings.get("hooks", {})
        assert "PreToolUse" in hooks, "PreToolUse hooks not configured"

    def test_posttooluse_hooks_configured(self):
        """Test that PostToolUse hooks are configured."""
        settings_path = Path(".claude/settings.json")
        content = settings_path.read_text()
        settings = json.loads(content)

        hooks = settings.get("hooks", {})
        assert "PostToolUse" in hooks, "PostToolUse hooks not configured"


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with hooks."""

    def test_circuit_breaker_module_exists(self):
        """Test that circuit breaker module exists."""
        circuit_breaker_path = Path("models/circuit_breaker.py")
        assert circuit_breaker_path.exists(), "circuit_breaker.py not found"

    def test_circuit_breaker_import(self):
        """Test that circuit breaker can be imported."""
        from models.circuit_breaker import TradingCircuitBreaker

        breaker = TradingCircuitBreaker()
        assert breaker is not None

    def test_circuit_breaker_halt_functionality(self):
        """Test circuit breaker halt functionality."""
        from models.circuit_breaker import TradingCircuitBreaker

        breaker = TradingCircuitBreaker()

        assert breaker.can_trade() is True

        # Trigger halt
        breaker.halt_trading(reason="Test halt")

        assert breaker.can_trade() is False


class TestComplianceIntegration:
    """Integration tests for compliance with hooks."""

    def test_audit_logger_integration(self):
        """Test audit logger integrates with hooks."""
        from compliance import create_audit_logger

        logger = create_audit_logger(auto_persist=False)

        # Log a trade action (simulating hook logging)
        entry = logger.log_trade(
            trade_id="TRD-001",
            symbol="SPY",
            quantity=100,
            price=450.0,
            side="buy",
        )

        assert entry.action == "TRADE_EXECUTED"
        assert entry.category.value == "trade"

    def test_anti_manipulation_integration(self):
        """Test anti-manipulation monitor integrates with hooks."""
        from datetime import datetime

        from compliance import create_anti_manipulation_monitor
        from compliance.anti_manipulation import OrderEvent

        monitor = create_anti_manipulation_monitor()

        # Process order event (simulating hook monitoring)
        event = OrderEvent(
            timestamp=datetime.utcnow(),
            order_id="ORD-001",
            symbol="SPY",
            side="buy",
            quantity=100,
            price=450.0,
            event_type="submitted",
        )

        alerts = monitor.process_order_event(event)
        assert isinstance(alerts, list)


class TestHookDirectory:
    """Integration tests for hook directory structure."""

    def test_hooks_directory_exists(self):
        """Test that .claude/hooks directory exists."""
        hooks_dir = Path(".claude/hooks")
        assert hooks_dir.exists(), ".claude/hooks directory not found"
        assert hooks_dir.is_dir()

    def test_hooks_directory_not_empty(self):
        """Test that hooks directory has files (in subdirectories)."""
        hooks_dir = Path(".claude/hooks")
        # Hooks are now in subdirectories
        hook_files = list(hooks_dir.glob("**/*.py"))
        assert len(hook_files) > 0, "No Python files in hooks directory"

    def test_all_hooks_have_valid_syntax(self):
        """Test that all hooks have valid Python syntax."""
        hooks_dir = Path(".claude/hooks")

        for hook_file in hooks_dir.glob("**/*.py"):
            code = hook_file.read_text()
            try:
                compile(code, str(hook_file), "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {hook_file}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
