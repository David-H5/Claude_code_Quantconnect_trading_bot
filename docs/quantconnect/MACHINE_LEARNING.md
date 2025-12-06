# Machine Learning on QuantConnect

Guide to integrating machine learning models with QuantConnect, including scikit-learn, TensorFlow, Keras, and model persistence.

## Table of Contents

- [Overview](#overview)
- [Supported Libraries](#supported-libraries)
- [Scikit-Learn Integration](#scikit-learn-integration)
- [TensorFlow Integration](#tensorflow-integration)
- [Keras Integration](#keras-integration)
- [Model Training Workflow](#model-training-workflow)
- [Model Persistence](#model-persistence)
- [Feature Engineering](#feature-engineering)
- [Best Practices](#best-practices)

## Overview

QuantConnect supports popular machine learning frameworks for building predictive models:

- **Scikit-Learn**: Ready-to-use algorithms (SVM, Random Forest, etc.)
- **TensorFlow**: Low-level deep learning framework
- **Keras**: High-level API for neural networks
- **XGBoost**: Gradient boosting

## Supported Libraries

> **2025 Update**: QuantConnect supports a wide range of ML libraries including PyTorch, Scikit-Learn, Stable Baselines, TensorFlow, Tslearn, XGBoost, and Hugging Face. Key package versions include: pytorch-lightning 2.5.2, pytorch-forecasting 1.4.0, TensorFlow 2.16.1, and OpenAI 1.14.3. Using incompatible library versions can lead to fatal errors. To request a new library, contact QuantConnect (2-4 week process).

```python
# White-listed ML libraries
import numpy as np
import pandas as pd
import scipy
import sklearn          # Scikit-learn
import tensorflow as tf # TensorFlow 2.16.1
import keras           # Keras (runs on TensorFlow)
import xgboost         # XGBoost
import torch           # PyTorch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
```

### Library Versions (as of 2025)

| Library | Version | Notes |
|---------|---------|-------|
| TensorFlow | 2.16.1 | Apache 2.0-licensed |
| PyTorch Lightning | 2.5.2 | High-level PyTorch |
| PyTorch Forecasting | 1.4.0 | Time series |
| OpenAI | 1.14.3 | LLM integration |
| Hugging Face | Supported | Transformers |

## Scikit-Learn Integration

Scikit-learn provides ready-to-use algorithms for classification and regression.

### Basic Classification Model

```python
from AlgorithmImports import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

class SKLearnAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Initialize model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

        # Lookback for features
        self.lookback = 20
        self.training_period = 252

        # Schedule training
        self.Schedule.On(
            self.DateRules.MonthStart(),
            self.TimeRules.AfterMarketOpen(self.symbol, 30),
            self.TrainModel
        )

        # Warm up
        self.SetWarmUp(self.training_period + self.lookback)

    def TrainModel(self):
        """Train the model on historical data."""
        # Get historical data
        history = self.History(self.symbol, self.training_period + self.lookback, Resolution.Daily)

        if history.empty:
            return

        # Prepare features
        X, y = self.PrepareData(history)

        if len(X) < 50:
            return

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Log training accuracy
        accuracy = self.model.score(X_scaled, y)
        self.Log(f"Model trained with accuracy: {accuracy:.2%}")

    def PrepareData(self, history):
        """Prepare features and labels."""
        df = history['close'].unstack(level=0)

        # Features: technical indicators
        features = []
        labels = []

        for i in range(self.lookback, len(df) - 1):
            window = df.iloc[i-self.lookback:i]

            # Calculate features
            returns = window.pct_change().dropna()
            feature = [
                returns.iloc[-1].values[0],      # Last return
                returns.mean().values[0],        # Mean return
                returns.std().values[0],         # Volatility
                returns.iloc[-5:].mean().values[0],  # 5-day momentum
                (window.iloc[-1] / window.iloc[0] - 1).values[0],  # Period return
            ]
            features.append(feature)

            # Label: 1 if next day return > 0
            next_return = (df.iloc[i+1] / df.iloc[i] - 1).values[0]
            labels.append(1 if next_return > 0 else 0)

        return np.array(features), np.array(labels)

    def OnData(self, data):
        if self.IsWarmingUp or not self.is_trained:
            return

        if not data.ContainsKey(self.symbol):
            return

        # Get recent data for prediction
        history = self.History(self.symbol, self.lookback + 1, Resolution.Daily)

        if history.empty:
            return

        # Prepare features for prediction
        df = history['close'].unstack(level=0)
        returns = df.pct_change().dropna()

        feature = [[
            returns.iloc[-1].values[0],
            returns.mean().values[0],
            returns.std().values[0],
            returns.iloc[-5:].mean().values[0],
            (df.iloc[-1] / df.iloc[0] - 1).values[0],
        ]]

        # Scale and predict
        feature_scaled = self.scaler.transform(feature)
        prediction = self.model.predict(feature_scaled)[0]
        probability = self.model.predict_proba(feature_scaled)[0]

        # Trade based on prediction
        if prediction == 1 and probability[1] > 0.6:
            if not self.Portfolio[self.symbol].Invested:
                self.SetHoldings(self.symbol, 0.95)
        elif prediction == 0 and probability[0] > 0.6:
            if self.Portfolio[self.symbol].Invested:
                self.Liquidate(self.symbol)
```

### Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

def Initialize(self):
    # SVM with RBF kernel
    self.model = SVC(kernel='rbf', C=1.0, probability=True)

    # Or linear SVM
    # self.model = SVC(kernel='linear', probability=True)
```

### Regression Model

```python
from sklearn.ensemble import GradientBoostingRegressor

class RegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.model = GradientBoostingRegressor(n_estimators=100)

    def TrainModel(self):
        # Prepare data
        X, y = self.PrepareData(history)

        # y is now continuous (next day return)
        self.model.fit(X, y)

    def OnData(self, data):
        # Predict expected return
        predicted_return = self.model.predict(feature)[0]

        # Trade based on predicted return
        if predicted_return > 0.005:  # > 0.5% expected return
            self.SetHoldings(self.symbol, 0.95)
        elif predicted_return < -0.005:
            self.Liquidate()
```

## TensorFlow Integration

TensorFlow provides building blocks for deep learning models.

```python
from AlgorithmImports import *
import tensorflow as tf
import numpy as np

class TensorFlowAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Build model
        self.model = self.BuildModel()
        self.is_trained = False

        self.Schedule.On(
            self.DateRules.MonthStart(),
            self.TimeRules.AfterMarketOpen(self.symbol, 30),
            self.TrainModel
        )

    def BuildModel(self):
        """Build TensorFlow model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def TrainModel(self):
        """Train the TensorFlow model."""
        history = self.History(self.symbol, 500, Resolution.Daily)

        if history.empty:
            return

        X, y = self.PrepareData(history)

        # Train with validation split
        self.model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        self.is_trained = True

        # Evaluate
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        self.Log(f"Model accuracy: {accuracy:.2%}")

    def PrepareData(self, history):
        """Prepare features with technical indicators."""
        df = history['close'].unstack(level=0).values.flatten()

        features = []
        labels = []

        for i in range(30, len(df) - 1):
            window = df[i-30:i]

            # Technical features
            feature = [
                (df[i] - df[i-1]) / df[i-1],         # 1-day return
                (df[i] - df[i-5]) / df[i-5],         # 5-day return
                (df[i] - df[i-10]) / df[i-10],       # 10-day return
                (df[i] - df[i-20]) / df[i-20],       # 20-day return
                np.std(window[-5:]),                  # 5-day volatility
                np.std(window[-20:]),                 # 20-day volatility
                np.mean(window[-5:]) / np.mean(window[-20:]) - 1,  # MA ratio
                (df[i] - np.min(window)) / (np.max(window) - np.min(window) + 1e-8),  # Position in range
                np.sum(np.diff(window[-5:]) > 0) / 4,  # Up days ratio
                np.mean(np.diff(window[-10:])),       # Trend strength
            ]
            features.append(feature)

            # Label: up day
            labels.append(1 if df[i+1] > df[i] else 0)

        return np.array(features), np.array(labels)

    def OnData(self, data):
        if self.IsWarmingUp or not self.is_trained:
            return

        # Get prediction
        history = self.History(self.symbol, 31, Resolution.Daily)
        df = history['close'].unstack(level=0).values.flatten()

        feature = self.GetLatestFeature(df)
        prediction = self.model.predict(np.array([feature]), verbose=0)[0][0]

        # Trade
        if prediction > 0.6:
            self.SetHoldings(self.symbol, 0.95)
        elif prediction < 0.4:
            self.Liquidate()

    def GetLatestFeature(self, df):
        """Get features for latest data point."""
        i = len(df) - 1
        window = df[i-30:i]

        return [
            (df[i] - df[i-1]) / df[i-1],
            (df[i] - df[i-5]) / df[i-5],
            (df[i] - df[i-10]) / df[i-10],
            (df[i] - df[i-20]) / df[i-20],
            np.std(window[-5:]),
            np.std(window[-20:]),
            np.mean(window[-5:]) / np.mean(window[-20:]) - 1,
            (df[i] - np.min(window)) / (np.max(window) - np.min(window) + 1e-8),
            np.sum(np.diff(window[-5:]) > 0) / 4,
            np.mean(np.diff(window[-10:])),
        ]
```

## Keras Integration

Keras provides a user-friendly API for building neural networks.

### LSTM for Time Series

```python
from AlgorithmImports import *
import tensorflow as tf
from tensorflow import keras
import numpy as np

class LSTMAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)

        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        self.sequence_length = 20
        self.model = self.BuildLSTMModel()

    def BuildLSTMModel(self):
        """Build LSTM model for sequence prediction."""
        model = keras.Sequential([
            keras.layers.LSTM(50, return_sequences=True,
                            input_shape=(self.sequence_length, 1)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(50, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(25),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def PrepareSequenceData(self, history):
        """Prepare sequence data for LSTM."""
        prices = history['close'].unstack(level=0).values.flatten()
        returns = np.diff(prices) / prices[:-1]

        X, y = [], []
        for i in range(self.sequence_length, len(returns) - 1):
            X.append(returns[i-self.sequence_length:i].reshape(-1, 1))
            y.append(1 if returns[i+1] > 0 else 0)

        return np.array(X), np.array(y)

    def TrainModel(self):
        history = self.History(self.symbol, 500, Resolution.Daily)
        X, y = self.PrepareSequenceData(history)

        self.model.fit(X, y, epochs=20, batch_size=32, verbose=0)
```

### Convolutional Neural Network

```python
def BuildCNNModel(self):
    """Build CNN for pattern recognition."""
    model = keras.Sequential([
        keras.layers.Conv1D(32, 3, activation='relu',
                          input_shape=(self.sequence_length, 1)),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(64, 3, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
```

## Model Training Workflow

### Research → Algorithm Pipeline

```python
# 1. Develop in Research Notebook
# research_notebook.ipynb
qb = QuantBook()
symbol = qb.add_equity("SPY").symbol
history = qb.history(symbol, 500, Resolution.DAILY)

# Train and evaluate model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
print(f"Test accuracy: {model.score(X_test, y_test)}")

# Save model to Object Store
import pickle
qb.object_store.save_bytes("trained_model.pkl", pickle.dumps(model))

# 2. Load in Algorithm
class MyAlgorithm(QCAlgorithm):
    def Initialize(self):
        model_bytes = self.ObjectStore.ReadBytes("trained_model.pkl")
        self.model = pickle.loads(model_bytes)
```

### Walk-Forward Training

```python
class WalkForwardMLAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.training_months = 12
        self.retraining_frequency = 1  # Monthly

        self.Schedule.On(
            self.DateRules.MonthStart(),
            self.TimeRules.AfterMarketOpen(self.symbol, 30),
            self.RetrainModel
        )

    def RetrainModel(self):
        """Walk-forward retraining."""
        # Use last N months for training
        history = self.History(
            self.symbol,
            self.training_months * 21,  # ~21 trading days per month
            Resolution.Daily
        )

        X, y = self.PrepareData(history)

        # Train on rolling window
        self.model.fit(X, y)
        self.Log(f"Model retrained at {self.Time}")
```

## Model Persistence

### Pickle for Scikit-Learn

```python
import pickle

def SaveModel(self):
    model_bytes = pickle.dumps(self.model)
    self.ObjectStore.SaveBytes("sklearn_model.pkl", model_bytes)

def LoadModel(self):
    if self.ObjectStore.ContainsKey("sklearn_model.pkl"):
        model_bytes = self.ObjectStore.ReadBytes("sklearn_model.pkl")
        self.model = pickle.loads(model_bytes)
```

### JSON for Keras/TensorFlow

```python
import json
import numpy as np

def SaveKerasModel(self):
    # Save architecture
    architecture = self.model.to_json()
    self.ObjectStore.Save("model_architecture.json", architecture)

    # Save weights as JSON (alternative to .h5)
    weights = [w.tolist() for w in self.model.get_weights()]
    self.ObjectStore.Save("model_weights.json", json.dumps(weights))

def LoadKerasModel(self):
    # Load architecture
    architecture = self.ObjectStore.Read("model_architecture.json")
    self.model = keras.models.model_from_json(architecture)

    # Load weights
    weights_json = self.ObjectStore.Read("model_weights.json")
    weights = [np.array(w) for w in json.loads(weights_json)]
    self.model.set_weights(weights)
```

## Feature Engineering

### Common Features for Trading

```python
def CalculateFeatures(self, history):
    """Calculate technical features."""
    df = history['close'].unstack(level=0)
    symbol_col = df.columns[0]

    features = pd.DataFrame(index=df.index)

    # Price-based
    features['return_1d'] = df[symbol_col].pct_change(1)
    features['return_5d'] = df[symbol_col].pct_change(5)
    features['return_20d'] = df[symbol_col].pct_change(20)

    # Volatility
    features['volatility_5d'] = features['return_1d'].rolling(5).std()
    features['volatility_20d'] = features['return_1d'].rolling(20).std()

    # Moving averages
    features['sma_ratio'] = df[symbol_col].rolling(5).mean() / df[symbol_col].rolling(20).mean()

    # RSI
    delta = df[symbol_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df[symbol_col].ewm(span=12).mean()
    ema26 = df[symbol_col].ewm(span=26).mean()
    features['macd'] = ema12 - ema26

    # Bollinger Band position
    sma20 = df[symbol_col].rolling(20).mean()
    std20 = df[symbol_col].rolling(20).std()
    features['bb_position'] = (df[symbol_col] - sma20) / (2 * std20)

    return features.dropna()
```

## Best Practices

### 1. Prevent Overfitting

```python
def TrainModel(self):
    X, y = self.PrepareData(history)

    # Use cross-validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(self.model, X, y, cv=5)
    self.Log(f"CV Score: {scores.mean():.2%} (+/- {scores.std():.2%})")

    # Train on full data
    self.model.fit(X, y)
```

### 2. Handle Memory Constraints

```python
def TrainModel(self):
    # Process data in chunks
    chunk_size = 1000

    history = self.History(self.symbol, 5000, Resolution.Daily)

    for i in range(0, len(history), chunk_size):
        chunk = history.iloc[i:i+chunk_size]
        X, y = self.PrepareData(chunk)

        # Partial fit for incremental learning
        self.model.partial_fit(X, y, classes=[0, 1])
```

### 3. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def Initialize(self):
    self.scaler = StandardScaler()
    # Or MinMaxScaler for neural networks

def TrainModel(self):
    X, y = self.PrepareData(history)

    # Fit scaler on training data
    X_scaled = self.scaler.fit_transform(X)
    self.model.fit(X_scaled, y)

def Predict(self, features):
    # Use same scaler for prediction
    features_scaled = self.scaler.transform(features)
    return self.model.predict(features_scaled)
```

### 4. Version Compatibility

```python
# Check library versions
import sklearn
import tensorflow as tf

self.Log(f"sklearn version: {sklearn.__version__}")
self.Log(f"TensorFlow version: {tf.__version__}")

# QuantConnect environment may have specific versions
# Test locally with matching versions
```

---

**Sources:**
- [Machine Learning Key Concepts](https://www.quantconnect.com/docs/v2/writing-algorithms/machine-learning/key-concepts)
- [Scikit-Learn Documentation](https://www.quantconnect.com/docs/v2/research-environment/machine-learning/scikit-learn)
- [TensorFlow Documentation](https://www.quantconnect.com/docs/v2/research-environment/machine-learning/tensorflow)
- [Keras Documentation](https://www.quantconnect.com/docs/v2/research-environment/machine-learning/keras)
- [Top ML Libraries Blog](https://www.quantconnect.com/blog/top-machine-learning-libraries-for-python/)

*Last Updated: November 2025*
