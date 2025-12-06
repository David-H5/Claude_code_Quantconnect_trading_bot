# Research Environment on QuantConnect

Guide to using the QuantConnect Research Environment for strategy development, data analysis, and machine learning.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [QuantBook API](#quantbook-api)
- [History Requests](#history-requests)
- [Indicators in Research](#indicators-in-research)
- [Data Analysis](#data-analysis)
- [Local Research Environment](#local-research-environment)
- [Best Practices](#best-practices)

## Overview

> **2025 Update**: The Research Environment now uses **Jupyter Lab** as the default interface (replacing classic Jupyter Notebook). QuantBook continues to serve as a thin wrapper around QCAlgorithm, providing access to all algorithm methods plus research-specific features like direct history requests and indicator warm-up.

The Research Environment provides Jupyter notebooks for:

- **Data Exploration**: Access historical market data
- **Strategy Development**: Test ideas before backtesting
- **Machine Learning**: Train and evaluate models
- **Visualization**: Create charts and analysis

### Research vs Backtesting

| Feature | Research | Backtesting |
|---------|----------|-------------|
| Data Access | All historical data | Only data at algorithm time |
| Event-Driven | No | Yes (OnData, etc.) |
| Time Simulation | No | Yes |
| Order Execution | No | Yes |
| Universe Selection | Manual | Automatic |

## Getting Started

### Creating a Research Notebook

```python
# First cell - import and initialize
from AlgorithmImports import *

# Create QuantBook instance (wrapper on QCAlgorithm)
qb = QuantBook()
```

### Basic Data Access

```python
# Add securities
spy = qb.AddEquity("SPY").Symbol
aapl = qb.AddEquity("AAPL").Symbol

# Get historical data
history = qb.History(spy, 360, Resolution.Daily)

# Display data
display(history.head())
```

## QuantBook API

QuantBook provides access to all QCAlgorithm methods plus research-specific features.

### Adding Securities

```python
# Equities
spy = qb.AddEquity("SPY", Resolution.Daily).Symbol
aapl = qb.AddEquity("AAPL", Resolution.Minute).Symbol

# Options
option = qb.AddOption("SPY", Resolution.Minute)
option.SetFilter(-5, 5, 30, 60)

# Futures
future = qb.AddFuture(Futures.Indices.SP500EMini, Resolution.Minute)
future.SetFilter(0, 90)

# Forex
eurusd = qb.AddForex("EURUSD", Resolution.Hour).Symbol

# Crypto
btc = qb.AddCrypto("BTCUSD", Resolution.Hour).Symbol
```

### Creating Indicators

```python
# Built-in indicators
rsi = qb.RSI(spy, 14, Resolution.Daily)
sma = qb.SMA(spy, 50, Resolution.Daily)
macd = qb.MACD(spy, 12, 26, 9, MovingAverageType.Exponential, Resolution.Daily)
bb = qb.BB(spy, 20, 2, Resolution.Daily)

# Warm up indicators
history = qb.History(spy, 100, Resolution.Daily)
for bar in history.itertuples():
    rsi.Update(bar.Index[1], bar.close)
    sma.Update(bar.Index[1], bar.close)

# Check readiness
print(f"RSI Ready: {rsi.IsReady}, Value: {rsi.Current.Value}")
```

## History Requests

### Basic History Request

```python
# By number of bars
history = qb.History(spy, 100, Resolution.Daily)

# By time period
history = qb.History(spy, timedelta(days=365), Resolution.Daily)

# By date range
history = qb.History(spy,
                     datetime(2023, 1, 1),
                     datetime(2023, 12, 31),
                     Resolution.Daily)
```

### Multiple Symbols

```python
# Add multiple symbols
symbols = [
    qb.AddEquity("SPY").Symbol,
    qb.AddEquity("QQQ").Symbol,
    qb.AddEquity("IWM").Symbol,
]

# Get history for all
history = qb.History(symbols, 252, Resolution.Daily)

# Access by symbol
spy_history = history.loc[symbols[0]]
```

### Return Types

```python
# DataFrame (default)
df = qb.History(spy, 100, Resolution.Daily)

# Access columns
closes = df['close']
volumes = df['volume']

# Slice objects
slices = qb.History[Slice](symbols, 100, Resolution.Daily)
for slice in slices:
    for symbol in slice.Keys:
        bar = slice.Bars[symbol]
        print(f"{symbol}: {bar.Close}")
```

### Different Data Types

```python
# Trade bars (OHLCV)
trade_bars = qb.History[TradeBar](spy, 100, Resolution.Daily)

# Quote bars (bid/ask)
quote_bars = qb.History[QuoteBar](spy, 100, Resolution.Minute)

# Tick data
ticks = qb.History[Tick](spy, timedelta(minutes=5), Resolution.Tick)
```

## Indicators in Research

### Using Indicators with History

```python
# Create indicator
rsi = RSI(14)

# Manual update with history
history = qb.History(spy, 100, Resolution.Daily)

rsi_values = []
for bar in history.itertuples():
    rsi.Update(bar.Index[1], bar.close)
    if rsi.IsReady:
        rsi_values.append(rsi.Current.Value)

# Plot RSI
import matplotlib.pyplot as plt
plt.plot(rsi_values)
plt.title('RSI')
plt.show()
```

### Indicator History Shortcut

```python
# Get indicator values directly
rsi = qb.RSI(spy, 14, Resolution.Daily)

# Get history as DataFrame
rsi_df = qb.Indicator(rsi, spy, 252, Resolution.Daily)

# Plot
rsi_df['RSI'].plot(title='RSI(14)')
```

### Multiple Indicators

```python
# Create multiple indicators
sma_fast = qb.SMA(spy, 10, Resolution.Daily)
sma_slow = qb.SMA(spy, 50, Resolution.Daily)

# Get history
history = qb.History(spy, 252, Resolution.Daily)

# Calculate indicator values
sma_fast_values = []
sma_slow_values = []

for bar in history.itertuples():
    sma_fast.Update(bar.Index[1], bar.close)
    sma_slow.Update(bar.Index[1], bar.close)

    if sma_fast.IsReady and sma_slow.IsReady:
        sma_fast_values.append(sma_fast.Current.Value)
        sma_slow_values.append(sma_slow.Current.Value)

# Compare
crossover_signals = [f > s for f, s in zip(sma_fast_values, sma_slow_values)]
```

## Data Analysis

### Calculating Returns

```python
import pandas as pd
import numpy as np

# Get history
history = qb.History(spy, 252, Resolution.Daily)

# Calculate returns
df = history['close'].unstack(level=0)
returns = df.pct_change().dropna()

# Summary statistics
print(f"Mean Daily Return: {returns.mean().values[0]:.4%}")
print(f"Daily Volatility: {returns.std().values[0]:.4%}")
print(f"Annualized Return: {returns.mean().values[0] * 252:.2%}")
print(f"Annualized Volatility: {returns.std().values[0] * np.sqrt(252):.2%}")
print(f"Sharpe Ratio: {(returns.mean().values[0] * 252) / (returns.std().values[0] * np.sqrt(252)):.2f}")
```

### Correlation Analysis

```python
# Multiple symbols
symbols = [qb.AddEquity(t).Symbol for t in ["SPY", "QQQ", "TLT", "GLD"]]
history = qb.History(symbols, 252, Resolution.Daily)

# Calculate returns
df = history['close'].unstack(level=0)
returns = df.pct_change().dropna()

# Correlation matrix
correlation = returns.corr()

# Plot heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

### Visualization

```python
import matplotlib.pyplot as plt

# Get data
history = qb.History(spy, 252, Resolution.Daily)
df = history['close'].unstack(level=0)

# Price chart
plt.figure(figsize=(12, 6))
plt.plot(df)
plt.title('SPY Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Multiple subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Price
axes[0].plot(df)
axes[0].set_title('Price')

# Returns
returns = df.pct_change()
axes[1].plot(returns)
axes[1].set_title('Daily Returns')

# Cumulative returns
cum_returns = (1 + returns).cumprod()
axes[2].plot(cum_returns)
axes[2].set_title('Cumulative Returns')

plt.tight_layout()
plt.show()
```

### Options Analysis

```python
# Add option
option = qb.AddOption("SPY")
option.SetFilter(-5, 5, 30, 60)

# Get option chain history
history = qb.OptionHistory(option.Symbol, datetime(2024, 1, 1), datetime(2024, 1, 31))

# Analyze
for chain in history:
    for contract in chain.Contracts.Values:
        print(f"Strike: {contract.Strike}")
        print(f"Expiry: {contract.Expiry}")
        print(f"IV: {contract.ImpliedVolatility:.2%}")
        print(f"Delta: {contract.Greeks.Delta:.2f}")
        print()
```

## Local Research Environment

### Running Locally with LEAN CLI

```bash
# Install LEAN CLI
pip install lean

# Initialize project
lean init my-project

# Launch research environment
lean research my-project
```

### Docker-Based Research

```bash
# Run research in Docker
lean research my-project --data-provider QuantConnect
```

### Local Data Provider

```bash
# Use local data
lean research my-project --data-provider Local

# Use QuantConnect data
lean research my-project --data-provider-historical QuantConnect
```

## Object Store in Research

### Saving Data

```python
# Save string
qb.ObjectStore.Save("my_key", "my_value")

# Save JSON
import json
data = {"model": "v1", "accuracy": 0.85}
qb.ObjectStore.Save("model_info", json.dumps(data))

# Save bytes (for ML models)
import pickle
model_bytes = pickle.dumps(my_model)
qb.ObjectStore.SaveBytes("model.pkl", model_bytes)
```

### Loading Data

```python
# Check existence
if qb.ObjectStore.ContainsKey("my_key"):
    value = qb.ObjectStore.Read("my_key")
    print(value)

# Load JSON
model_info = json.loads(qb.ObjectStore.Read("model_info"))

# Load model
model_bytes = qb.ObjectStore.ReadBytes("model.pkl")
my_model = pickle.loads(model_bytes)
```

## Best Practices

### 1. Start with Research

```python
# 1. Explore data in research
history = qb.History(spy, 500, Resolution.Daily)

# 2. Test indicators
rsi = qb.RSI(spy, 14, Resolution.Daily)

# 3. Develop strategy logic
# ...

# 4. Move to backtest
# Copy working code to algorithm
```

### 2. Use Efficient Data Requests

```python
# GOOD: Single request for all symbols
symbols = [qb.AddEquity(t).Symbol for t in tickers]
history = qb.History(symbols, 252, Resolution.Daily)

# BAD: Multiple individual requests
for ticker in tickers:
    symbol = qb.AddEquity(ticker).Symbol
    history = qb.History(symbol, 252, Resolution.Daily)  # Slow!
```

### 3. Handle Missing Data

```python
history = qb.History(spy, 252, Resolution.Daily)

if history.empty:
    print("No data available")
else:
    # Process data
    pass

# Check for NaN
if history.isnull().any().any():
    history = history.fillna(method='ffill')
```

### 4. Memory Management

```python
# For large datasets, process in chunks
total_days = 2520  # 10 years
chunk_size = 252

all_results = []

for i in range(0, total_days, chunk_size):
    start = datetime(2014, 1, 1) + timedelta(days=i)
    end = start + timedelta(days=chunk_size)

    chunk = qb.History(spy, start, end, Resolution.Daily)
    result = process_chunk(chunk)
    all_results.append(result)

    # Clear memory
    del chunk

final_result = pd.concat(all_results)
```

### 5. Document Your Analysis

```python
# Use markdown cells for documentation

# Hypothesis:
# Moving average crossover strategies work better in trending markets

# Test Period: 2020-2023
# Assets: SPY, QQQ, IWM

# Results:
# SPY: Sharpe 1.2, Max DD 15%
# QQQ: Sharpe 1.4, Max DD 18%
# IWM: Sharpe 0.9, Max DD 20%

# Conclusion:
# Strategy works best on SPY and QQQ
```

### 6. Save Results for Algorithms

```python
# Train model in research
from sklearn.ensemble import RandomForestClassifier

X, y = prepare_data(history)
model = RandomForestClassifier()
model.fit(X, y)

# Save to Object Store
import pickle
qb.ObjectStore.SaveBytes("rf_model.pkl", pickle.dumps(model))

# Now algorithm can load it
# self.ObjectStore.ReadBytes("rf_model.pkl")
```

---

**Sources:**
- [Research Environment](https://www.quantconnect.com/docs/v2/research-environment)
- [Research Engine](https://www.quantconnect.com/docs/v2/research-environment/key-concepts/research-engine)
- [Datasets Key Concepts](https://www.quantconnect.com/docs/v2/research-environment/datasets/key-concepts)
- [lean research CLI](https://www.quantconnect.com/docs/v2/lean-cli/api-reference/lean-research)
- [Storing Data](https://www.quantconnect.com/docs/v2/research-environment/tutorials/storing-data)

*Last Updated: November 2025*
