# Run Backtest

Run a backtest for the specified algorithm using LEAN CLI.

## Arguments
- `$ARGUMENTS` - Algorithm file path or name (e.g., "simple_momentum" or "algorithms/simple_momentum.py")

## Instructions

1. Verify the algorithm file exists in `algorithms/` directory
2. Check that LEAN CLI is installed and configured
3. Run the backtest command:
   ```bash
   lean backtest "algorithms/$ARGUMENTS.py"
   ```
4. If LEAN is not available, provide instructions for running on QuantConnect cloud:
   - Upload the algorithm to quantconnect.com
   - Configure backtest parameters
   - Run and analyze results

5. After backtest completion, summarize:
   - Total return
   - Sharpe ratio
   - Max drawdown
   - Number of trades
   - Win rate (if available)

Algorithm to backtest: $ARGUMENTS
