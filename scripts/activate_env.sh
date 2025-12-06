#!/bin/bash
# Activate the project virtual environment
# Usage: source scripts/activate_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✅ Virtual environment activated"
    echo "   Python: $(python --version)"
    echo "   Pip: $(pip --version | cut -d' ' -f1-2)"
    echo ""
    echo "Available commands:"
    echo "   pytest tests/           - Run all tests"
    echo "   pytest tests/ -v        - Run tests with verbose output"
    echo "   pre-commit run --all    - Run all pre-commit hooks"
    echo "   ruff check .            - Run Ruff linter"
    echo "   mypy .                  - Run type checker"
    echo "   deactivate              - Exit virtual environment"
else
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "   Run: python3 -m venv .venv"
    echo "   Then: pip install -r requirements.txt"
fi
