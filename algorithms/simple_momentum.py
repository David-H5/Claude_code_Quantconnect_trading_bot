"""
Simple Momentum Strategy

A momentum-based trading strategy using RSI indicator to identify oversold
and overbought conditions.

Strategy Logic:
- Buy when RSI crosses above oversold threshold (default: 30)
- Sell when RSI crosses above overbought threshold (default: 70)

Author: Your Name
Date: 2025-11-24
"""

from AlgorithmImports import *


class SimpleMomentumAlgorithm(QCAlgorithm):
    """
    Momentum strategy using RSI indicator for SPY trading.

    Attributes:
        symbol: Trading symbol (SPY)
        rsi: Relative Strength Index indicator
        rsi_period: Lookback period for RSI calculation
        oversold_threshold: RSI level considered oversold
        overbought_threshold: RSI level considered overbought
    """

    def Initialize(self) -> None:
        """
        Initialize algorithm settings and RSI indicator.
        """
        # Set backtest period
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)

        # Set starting capital
        self.SetCash(100000)

        # Subscribe to SPY daily data
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Strategy parameters
        self.rsi_period = 14
        self.oversold_threshold = 30
        self.overbought_threshold = 70

        # Create RSI indicator
        self.rsi = self.RSI(self.symbol, self.rsi_period, Resolution.Daily)

        # Warm up the indicator
        self.SetWarmUp(self.rsi_period)

        # Set benchmark
        self.SetBenchmark("SPY")

    def OnData(self, data: Slice) -> None:
        """
        Process market data and execute trades based on RSI signals.

        Args:
            data: Slice object containing market data
        """
        # Skip during warmup period
        if self.IsWarmingUp:
            return

        # Ensure we have data and indicator is ready
        if not data.ContainsKey(self.symbol) or not self.rsi.IsReady:
            return

        rsi_value = self.rsi.Current.Value
        holdings = self.Portfolio[self.symbol]

        # Entry signal: RSI crosses above oversold threshold
        if rsi_value < self.oversold_threshold and not holdings.Invested:
            self.SetHoldings(self.symbol, 1.0)
            self.Debug(f"BUY: RSI = {rsi_value:.2f}")

        # Exit signal: RSI crosses above overbought threshold
        elif rsi_value > self.overbought_threshold and holdings.Invested:
            self.Liquidate(self.symbol)
            self.Debug(f"SELL: RSI = {rsi_value:.2f}")

    def OnEndOfAlgorithm(self) -> None:
        """
        Called at the end of the algorithm execution.
        Log final portfolio statistics.
        """
        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
