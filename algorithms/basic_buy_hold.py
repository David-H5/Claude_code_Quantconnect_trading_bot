"""
Basic Buy and Hold Algorithm

A simple starter algorithm that buys SPY at the beginning and holds it throughout
the backtest period. This serves as a baseline for comparing more complex strategies.

Author: Your Name
Date: 2025-11-24
"""

from AlgorithmImports import *


class BasicBuyAndHoldAlgorithm(QCAlgorithm):
    """
    Simple buy-and-hold strategy for SPY ETF.

    This algorithm demonstrates the basic structure of a QuantConnect algorithm:
    - Initialize: Set up algorithm parameters
    - OnData: Handle incoming market data
    """

    def Initialize(self) -> None:
        """
        Initialize algorithm settings, data subscriptions, and parameters.
        """
        # Set backtest period
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)

        # Set starting capital
        self.SetCash(100000)

        # Subscribe to SPY daily data
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Flag to track if we've made initial purchase
        self.purchased = False

        # Optional: Set benchmark
        self.SetBenchmark("SPY")

    def OnData(self, data: Slice) -> None:
        """
        Process incoming market data and execute trades.

        Args:
            data: Slice object containing market data for subscribed securities
        """
        # Buy on first data point
        if not self.purchased:
            if data.ContainsKey(self.symbol):
                # Invest 100% of portfolio in SPY
                self.SetHoldings(self.symbol, 1.0)
                self.purchased = True
                self.Debug(f"Purchased SPY at {data[self.symbol].Close}")
