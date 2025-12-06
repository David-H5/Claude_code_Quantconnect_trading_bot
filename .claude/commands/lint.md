# Lint and Format Code

Run code quality checks and formatting on the codebase.

## Arguments
- `$ARGUMENTS` - Optional: specific file or directory to check (leave empty for all)

## Instructions

1. Run Black formatter (check mode first):
   ```bash
   black --check --line-length 100 algorithms/ indicators/ models/ utils/ tests/
   ```

2. Run Flake8 linter:
   ```bash
   flake8 algorithms/ indicators/ models/ utils/ tests/ --max-line-length=100
   ```

3. Run MyPy type checker:
   ```bash
   mypy algorithms/ indicators/ models/ utils/ --ignore-missing-imports
   ```

4. If issues found:
   - List all issues by severity
   - Offer to auto-fix formatting with Black
   - Provide guidance on fixing linting errors
   - Suggest type hint additions where missing

5. Report summary:
   - Total issues found
   - Issues by category (formatting, linting, typing)
   - Files with most issues

Target: $ARGUMENTS
