# Create New Trading Algorithm

Create a new QuantConnect trading algorithm with the specified strategy.

## Arguments
- `$ARGUMENTS` - Strategy description (e.g., "MACD crossover strategy for tech stocks")

## Instructions

1. Create a new algorithm file in `algorithms/` directory
2. Use the naming convention: `snake_case.py` (e.g., `macd_crossover.py`)
3. Follow the algorithm template structure:
   - Module docstring with strategy description
   - Class inheriting from `QCAlgorithm`
   - Proper `Initialize()` with all settings
   - `OnData()` with trading logic
   - Include risk management
   - Add proper logging

4. The algorithm must include:
   - Type hints on all methods
   - Google-style docstrings
   - Configurable parameters as class attributes
   - Proper indicator warmup
   - Data validation checks
   - Debug logging for entry/exit signals

5. Create a corresponding test file in `tests/` directory

6. Update `algorithms/__init__.py` to export the new algorithm

Strategy to implement: $ARGUMENTS
