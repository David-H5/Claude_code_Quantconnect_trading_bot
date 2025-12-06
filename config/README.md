# Configuration Files

This directory contains configuration files for algorithms and backtests.

## Structure

- `algorithm_configs/`: Parameter settings for different algorithms
- `backtest_configs/`: Backtest settings (dates, capital, etc.)
- `live_configs/`: Live trading configurations

## Usage

Create JSON or YAML configuration files to make algorithms more flexible:

```json
{
  "strategy_name": "momentum_strategy",
  "parameters": {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "position_size": 0.1
  },
  "backtest": {
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "initial_cash": 100000
  }
}
```

## Best Practices

- Keep sensitive data (API keys) in environment variables, not config files
- Use different configs for different market conditions
- Document what each parameter does
- Version control config files
