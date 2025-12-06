# Run Tests

Run the test suite with coverage reporting.

## Arguments
- `$ARGUMENTS` - Optional: specific test file or test pattern (leave empty for all tests)

## Instructions

1. Run pytest with coverage:
   ```bash
   pytest tests/ -v --cov=algorithms --cov=indicators --cov=models --cov=utils --cov-report=term-missing
   ```

2. If specific tests requested:
   ```bash
   pytest tests/$ARGUMENTS -v
   ```

3. Report:
   - Number of tests passed/failed
   - Coverage percentage
   - Any uncovered lines that should be tested
   - Suggestions for improving test coverage

Test target: $ARGUMENTS
