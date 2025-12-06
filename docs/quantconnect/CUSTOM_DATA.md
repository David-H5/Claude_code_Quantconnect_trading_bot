# Custom Data and Alternative Data on QuantConnect

Guide to importing custom data sources, using alternative data, and integrating external data into your algorithms.

## Table of Contents

- [Overview](#overview)
- [Alternative Data Sources](#alternative-data-sources)
- [Custom Data Implementation](#custom-data-implementation)
- [File Providers](#file-providers)
- [Object Store](#object-store)
- [Data Formats](#data-formats)
- [Best Practices](#best-practices)

## Overview

QuantConnect provides multiple ways to incorporate data beyond standard market data:

1. **Alternative Data**: Pre-integrated data sources (SEC filings, news, sentiment)
2. **Custom Data**: User-defined data types from external sources
3. **Object Store**: Persistent storage for models and state

## Alternative Data Sources

QuantConnect offers a library of alternative data sources through the Data Market.

### Available Alternative Data

| Category | Examples |
|----------|----------|
| **SEC Filings** | 10-K, 10-Q, 8-K reports |
| **News & Sentiment** | Tiingo News, Brain Sentiment, Benzinga |
| **Economic Data** | Federal Reserve (FRED), Treasury rates, EIA |
| **Corporate Actions** | Dividends, splits, earnings, Smart Insider Buybacks |
| **Social Media** | Quiver WallStreetBets, Twitter sentiment |
| **Options Data** | Unusual options activity |
| **Political/Lobbying** | Quiver US Congress Trading, Corporate Lobbying |
| **Cryptocurrency** | CoinGecko Market Cap |

> **2025 Update**: Tiingo News Feed covers 10,000+ US Equities with data from 120+ news providers, starting January 2014 with second-frequency delivery. Brain Sentiment Indicator provides NLP-based sentiment from financial news sources. Most alternative datasets update daily/hourly; some (Tiingo, Benzinga) include live streams.

### Using Alternative Data

```python
from AlgorithmImports import *
from QuantConnect.DataSource import *

class AlternativeDataAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # Add equity
        self.symbol = self.AddEquity("AAPL", Resolution.Daily).Symbol

        # Add Tiingo News
        self.tiingo = self.AddData(TiingoNews, self.symbol).Symbol

        # Add SEC 8-K filings
        self.sec_8k = self.AddData(SEC8KReport, self.symbol).Symbol

    def OnData(self, data):
        # Access news data
        if data.ContainsKey(self.tiingo):
            news = data[self.tiingo]
            self.Log(f"News: {news.Title}")
            self.Log(f"Sentiment: {news.Sentiment}")

        # Access SEC filings
        if data.ContainsKey(self.sec_8k):
            filing = data[self.sec_8k]
            self.Log(f"SEC 8-K Filed: {filing.FormType}")
```

### Namespace Changes (2024)

Starting with Lean v2.5.12525, alternative data types are under `QuantConnect.DataSource`:

```python
# Old import (deprecated)
# from QuantConnect.Data.Custom import TiingoNews

# New import
from QuantConnect.DataSource import TiingoNews
```

## Custom Data Implementation

Custom data allows you to define any data source with a TIME and VALUE.

### Basic Custom Data Class

```python
from AlgorithmImports import *

class MyCustomData(PythonData):
    """Custom data type for external CSV data."""

    def GetSource(self, config, date, isLiveMode):
        """
        Return the URL to fetch data from.

        Called once per OnData cycle based on resolution.
        """
        # Static URL
        return SubscriptionDataSource(
            "https://example.com/data.csv",
            SubscriptionTransportMedium.RemoteFile
        )

        # Or dynamic URL based on date
        # url = f"https://example.com/data/{date.strftime('%Y-%m-%d')}.csv"
        # return SubscriptionDataSource(url, SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):
        """
        Parse each line of data into an object.

        Called for each line in the data file.
        """
        if not line or line.startswith('#'):
            return None

        data = MyCustomData()
        data.Symbol = config.Symbol

        try:
            parts = line.split(',')
            data.Time = datetime.strptime(parts[0], "%Y-%m-%d")
            data.EndTime = data.Time + timedelta(days=1)
            data.Value = float(parts[1])
            data["Signal"] = float(parts[2])  # Custom property
        except:
            return None

        return data

# Usage in algorithm
class CustomDataAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # Add custom data
        self.custom_symbol = self.AddData(MyCustomData, "MY_DATA").Symbol

    def OnData(self, data):
        if data.ContainsKey(self.custom_symbol):
            custom = data[self.custom_symbol]
            self.Log(f"Value: {custom.Value}, Signal: {custom['Signal']}")
```

### JSON Custom Data

```python
class JsonCustomData(PythonData):
    """Custom data from JSON API."""

    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource(
            "https://api.example.com/data.json",
            SubscriptionTransportMedium.RemoteFile,
            FileFormat.UnfoldingCollection  # For JSON arrays
        )

    def Reader(self, config, line, date, isLiveMode):
        if not line:
            return None

        data = JsonCustomData()
        data.Symbol = config.Symbol

        try:
            obj = json.loads(line)
            data.Time = datetime.strptime(obj['date'], "%Y-%m-%d")
            data.Value = float(obj['price'])
            data["volume"] = int(obj['volume'])
            data["sentiment"] = float(obj.get('sentiment', 0))
        except:
            return None

        return data
```

### REST API Custom Data

```python
class RestApiData(PythonData):
    """Custom data from REST API with authentication."""

    API_KEY = "your-api-key"

    def GetSource(self, config, date, isLiveMode):
        url = f"https://api.example.com/data?date={date.strftime('%Y-%m-%d')}"

        # Add headers for authentication
        headers = [("Authorization", f"Bearer {self.API_KEY}")]

        return SubscriptionDataSource(
            url,
            SubscriptionTransportMedium.Rest,
            FileFormat.Csv,
            headers
        )

    def Reader(self, config, line, date, isLiveMode):
        # Parse response
        pass
```

## File Providers

### Object Store (Recommended)

The Object Store is the fastest and most reliable file provider.

```python
class ObjectStoreDataAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        # Save data to object store
        data = "date,value\n2024-01-01,100\n2024-01-02,102"
        self.ObjectStore.Save("my_data.csv", data)

        # Add custom data from object store
        self.AddData(ObjectStoreData, "MY_DATA")

class ObjectStoreData(PythonData):

    def GetSource(self, config, date, isLiveMode):
        # Use object store path
        return SubscriptionDataSource(
            "my_data.csv",
            SubscriptionTransportMedium.ObjectStore
        )

    def Reader(self, config, line, date, isLiveMode):
        # Parse line
        pass
```

### Dropbox

```python
def GetSource(self, config, date, isLiveMode):
    # Dropbox shared link (add ?dl=1 for direct download)
    url = "https://www.dropbox.com/s/abc123/data.csv?dl=1"
    return SubscriptionDataSource(url, SubscriptionTransportMedium.RemoteFile)
```

**Note**: Dropbox caps download speeds to 10 kb/s after 3-4 requests.

### GitHub

```python
def GetSource(self, config, date, isLiveMode):
    # GitHub raw content URL
    url = "https://raw.githubusercontent.com/user/repo/main/data.csv"
    return SubscriptionDataSource(url, SubscriptionTransportMedium.RemoteFile)
```

### Google Sheets

```python
def GetSource(self, config, date, isLiveMode):
    # Google Sheets published as CSV
    sheet_id = "your-sheet-id"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    return SubscriptionDataSource(url, SubscriptionTransportMedium.RemoteFile)
```

## Object Store

The Object Store provides persistent key-value storage for your algorithms.

> **2025 Update**: Object Store provides **50MB free storage** per organization. For live trading algorithms, individual objects should remain under 50MB for optimal performance. Use `SaveBytes`/`ReadBytes` for binary data (ML models) and `SaveJson`/`ReadJson` for structured data. Objects persist across backtests and live deployments.

### Saving Data

```python
def Initialize(self):
    # Save string
    self.ObjectStore.Save("key", "string value")
    self.ObjectStore.SaveString("key", "string value")

    # Save JSON
    data = {"model": "v1", "params": [1, 2, 3]}
    self.ObjectStore.SaveJson("config", data)

    # Save bytes (for models)
    model_bytes = pickle.dumps(my_model)
    self.ObjectStore.SaveBytes("model.pkl", model_bytes)
```

### Reading Data

```python
def Initialize(self):
    # Check if key exists
    if self.ObjectStore.ContainsKey("key"):
        # Read string
        value = self.ObjectStore.Read("key")

        # Read JSON
        config = self.ObjectStore.ReadJson("config")

        # Read bytes
        model_bytes = self.ObjectStore.ReadBytes("model.pkl")
        my_model = pickle.loads(model_bytes)
```

### ML Model Persistence

```python
import pickle
from sklearn.ensemble import RandomForestClassifier

class MLModelAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Try to load existing model
        if self.ObjectStore.ContainsKey("rf_model.pkl"):
            model_bytes = self.ObjectStore.ReadBytes("rf_model.pkl")
            self.model = pickle.loads(model_bytes)
            self.Log("Loaded existing model")
        else:
            self.model = None
            self.Log("No existing model found")

        # Schedule model training
        self.Schedule.On(
            self.DateRules.MonthStart(),
            self.TimeRules.AfterMarketOpen(self.symbol, 60),
            self.TrainModel
        )

    def TrainModel(self):
        """Train and save model."""
        # Get training data
        history = self.History(self.symbol, 252, Resolution.Daily)

        # Prepare features and labels
        X, y = self.PrepareData(history)

        # Train model
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X, y)

        # Save model to object store
        model_bytes = pickle.dumps(self.model)
        self.ObjectStore.SaveBytes("rf_model.pkl", model_bytes)
        self.Log("Model trained and saved")

    def PrepareData(self, history):
        # Feature engineering logic
        pass
```

### TensorFlow Model Storage

```python
import tensorflow as tf
import json

class TensorFlowAlgorithm(QCAlgorithm):

    def Initialize(self):
        # Load model weights from JSON (alternative to .h5)
        if self.ObjectStore.ContainsKey("model_weights.json"):
            weights_json = self.ObjectStore.Read("model_weights.json")
            weights = json.loads(weights_json)
            self.model = self.BuildModel()
            self.model.set_weights([np.array(w) for w in weights])
        else:
            self.model = self.BuildModel()

    def BuildModel(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def SaveModel(self):
        # Save weights as JSON
        weights = [w.tolist() for w in self.model.get_weights()]
        self.ObjectStore.Save("model_weights.json", json.dumps(weights))
```

## Data Formats

### CSV Format

```python
class CsvData(PythonData):

    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource(
            "https://example.com/data.csv",
            SubscriptionTransportMedium.RemoteFile,
            FileFormat.Csv
        )

    def Reader(self, config, line, date, isLiveMode):
        if not line or line.startswith('date'):  # Skip header
            return None

        data = CsvData()
        data.Symbol = config.Symbol

        parts = line.split(',')
        data.Time = datetime.strptime(parts[0], "%Y-%m-%d")
        data.Value = float(parts[1])

        return data
```

### JSON Array Format

```python
class JsonArrayData(PythonData):

    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource(
            "https://api.example.com/data.json",
            SubscriptionTransportMedium.RemoteFile,
            FileFormat.UnfoldingCollection  # Unfolds JSON array
        )

    def Reader(self, config, line, date, isLiveMode):
        # Each element of JSON array is passed as 'line'
        obj = json.loads(line)

        data = JsonArrayData()
        data.Symbol = config.Symbol
        data.Time = datetime.fromisoformat(obj['timestamp'])
        data.Value = float(obj['value'])

        return data
```

## Best Practices

### 1. Handle Missing Data

```python
def Reader(self, config, line, date, isLiveMode):
    try:
        data = MyCustomData()
        parts = line.split(',')

        # Validate data
        if len(parts) < 3:
            return None

        data.Time = datetime.strptime(parts[0], "%Y-%m-%d")
        data.Value = float(parts[1]) if parts[1] else 0

        return data
    except Exception as e:
        # Return None for invalid data
        return None
```

### 2. Cache Data Locally

```python
class CachedData(PythonData):
    """Use Object Store as cache."""

    def GetSource(self, config, date, isLiveMode):
        cache_key = f"data_{date.strftime('%Y%m%d')}"

        # Check cache first
        if config.ObjectStore.ContainsKey(cache_key):
            return SubscriptionDataSource(
                cache_key,
                SubscriptionTransportMedium.ObjectStore
            )

        # Otherwise fetch from remote
        return SubscriptionDataSource(
            f"https://api.example.com/data/{date.strftime('%Y-%m-%d')}",
            SubscriptionTransportMedium.RemoteFile
        )
```

### 3. Rate Limit Handling

```python
import time

class RateLimitedData(PythonData):
    last_request_time = datetime.min
    min_request_interval = timedelta(seconds=1)

    def GetSource(self, config, date, isLiveMode):
        # Ensure minimum interval between requests
        elapsed = datetime.now() - RateLimitedData.last_request_time
        if elapsed < RateLimitedData.min_request_interval:
            time.sleep((RateLimitedData.min_request_interval - elapsed).total_seconds())

        RateLimitedData.last_request_time = datetime.now()

        return SubscriptionDataSource(
            "https://api.example.com/data",
            SubscriptionTransportMedium.RemoteFile
        )
```

### 4. Validate Data Quality

```python
def OnData(self, data):
    if not data.ContainsKey(self.custom_symbol):
        return

    custom = data[self.custom_symbol]

    # Validate before using
    if custom.Value <= 0:
        self.Log(f"Invalid value: {custom.Value}")
        return

    if custom.Time > self.Time:
        self.Log(f"Future data detected: {custom.Time}")
        return

    # Use validated data
    self.ProcessData(custom)
```

---

**Sources:**
- [Importing Data Key Concepts](https://www.quantconnect.com/docs/v2/writing-algorithms/importing-data/key-concepts)
- [Custom Data LEAN CLI](https://www.quantconnect.com/docs/v2/lean-cli/datasets/custom-data)
- [Object Store](https://www.quantconnect.com/docs/v2/writing-algorithms/object-store)
- [Storing Data](https://www.quantconnect.com/docs/v2/research-environment/tutorials/storing-data)

*Last Updated: November 2025*
