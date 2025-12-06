# QuantConnect Object Store Guide (5GB Tier)

Complete guide to using QuantConnect's Object Store for persistent data storage in your options trading bot.

**Tier:** 5GB
**Cost:** $20/month
**File Limit:** 50,000 files
**Max File Size:** 50MB (45MB recommended with compression)

---

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Usage Patterns](#usage-patterns)
- [Storage Categories](#storage-categories)
- [Monitoring & Alerts](#monitoring--alerts)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What is Object Store?

QuantConnect's Object Store is a key-value storage system for persisting data across:
- Backtest runs
- Research → Live trading workflows
- Algorithm restarts
- Organization-wide access

### 5GB Tier Specifications

| Feature | Specification |
|---------|--------------|
| **Storage** | 5GB |
| **Files** | 50,000 max |
| **File Size** | <50MB per file |
| **Cost** | $20/month |
| **Shared** | Across entire organization |
| **Access** | All projects in organization |

### Why 5GB?

Perfect for your trading bot needs:
- ✅ Trading state persistence (500MB)
- ✅ Options Greeks history (1.5GB)
- ✅ LLM analysis cache (300MB)
- ✅ Backtest results archive (500MB)
- ✅ ML models storage (1.5GB)
- ✅ Monitoring data (300MB)
- ✅ Research datasets (400MB)

**Total Allocated:** 5.0GB

---

## Configuration

### Settings ([config/settings.json](../../config/settings.json))

```json
{
  "quantconnect": {
    "object_store": {
      "enabled": true,
      "tier": "5GB",
      "max_storage_gb": 5,
      "max_files": 50000,
      "max_file_size_mb": 45,
      "compression_enabled": true,
      "auto_cleanup": {
        "enabled": true,
        "keep_days": 90,
        "archive_before_delete": true
      },
      "storage_allocation": {
        "trading_state": 0.5,
        "options_greeks": 1.5,
        "llm_cache": 0.3,
        "backtest_results": 0.5,
        "ml_models": 1.5,
        "monitoring_data": 0.3,
        "research_data": 0.4
      },
      "usage_monitoring": {
        "enabled": true,
        "alert_threshold_pct": 80,
        "critical_threshold_pct": 90,
        "check_interval_hours": 6
      }
    }
  }
}
```

---

## Usage Patterns

### 1. Trading State Persistence

Save daily trading state automatically:

```python
# Automatically saved in algorithm
def _save_trading_state(self):
    state = {
        "timestamp": str(self.Time),
        "equity": self.Portfolio.TotalPortfolioValue,
        "positions": {...},
        "circuit_breaker": self.circuit_breaker.get_status(),
    }

    self.object_store_manager.save(
        key=f"trading_state_{self.Time.date()}",
        data=state,
        category=StorageCategory.TRADING_STATE,
        expire_days=90,
    )
```

**Storage:** ~5MB/day × 90 days = **450MB**

### 2. Options Greeks History

Track underpriced opportunities:

```python
def save_greeks_snapshot(self, contract, greeks):
    key = f"greeks_{contract.Symbol}_{self.Time.date()}"

    data = {
        "symbol": str(contract.Symbol),
        "delta": greeks.Delta,
        "gamma": greeks.Gamma,
        "theta": greeks.Theta,
        "vega": greeks.Vega,
        "iv": contract.ImpliedVolatility,
        "price": contract.LastPrice,
        "timestamp": str(self.Time),
    }

    self.object_store_manager.save(
        key=key,
        data=data,
        category=StorageCategory.OPTIONS_GREEKS,
        expire_days=180,
    )
```

**Storage:** ~10MB/day × 180 days = **1.8GB** (compressed to ~900MB)

### 3. LLM Sentiment Cache

Avoid re-analyzing same news:

```python
def cache_sentiment(self, headline, sentiment_result):
    # Hash headline for key
    key = f"sentiment_{hash(headline)}"

    # Check if already cached
    cached = self.object_store_manager.load(key)
    if cached:
        return cached

    # Cache new result
    data = {
        "headline": headline,
        "sentiment": sentiment_result.sentiment.signal.name,
        "confidence": sentiment_result.sentiment.confidence,
        "score": sentiment_result.sentiment.score,
    }

    self.object_store_manager.save(
        key=key,
        data=data,
        category=StorageCategory.LLM_CACHE,
        expire_days=30,  # News gets stale
    )

    return data
```

**Storage:** ~10KB/headline × 30,000 headlines = **300MB**

### 4. Backtest Results Archive

Store parameter optimization results:

```python
def save_backtest_results(self, params, metrics):
    key = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    results = {
        "params": params,
        "metrics": {
            "sharpe_ratio": metrics.sharpe,
            "max_drawdown": metrics.max_drawdown,
            "total_return": metrics.total_return,
            "win_rate": metrics.win_rate,
        },
        "trades": metrics.trades,
    }

    self.object_store_manager.save(
        key=key,
        data=results,
        category=StorageCategory.BACKTEST_RESULTS,
        expire_days=365,
    )
```

**Storage:** ~5MB/backtest × 100 backtests = **500MB**

### 5. ML Models Storage

Store trained models for live deployment:

```python
# In research notebook
import pickle
import base64

# Train model
model = train_sentiment_model(data)

# Serialize and compress
model_bytes = pickle.dumps(model)
compressed = gzip.compress(model_bytes)
encoded = base64.b64encode(compressed).decode('utf-8')

# Save to Object Store
qb.ObjectStore.Save("sentiment_model_v2", encoded)

# In live algorithm
def load_ml_model(self):
    if self.ObjectStore.ContainsKey("sentiment_model_v2"):
        encoded = self.ObjectStore.Read("sentiment_model_v2")
        compressed = base64.b64decode(encoded.encode('utf-8'))
        model_bytes = gzip.decompress(compressed)
        model = pickle.loads(model_bytes)
        return model
```

**Storage:** ~400MB/model × 3 models = **1.2GB**

### 6. Research ↔ Live Transfer

Optimized parameters from research to live:

```python
# Research: Find optimal parameters
optimized = {
    "profit_thresholds": [1.0, 2.0, 4.0, 10.0],
    "cancel_after_seconds": 2.5,
    "max_option_chains": 5,
    "delta_range": [0.25, 0.35],
}

qb.ObjectStore.Save("optimized_params_v3", json.dumps(optimized))

# Live: Load and apply
if self.ObjectStore.ContainsKey("optimized_params_v3"):
    params = json.loads(self.ObjectStore.Read("optimized_params_v3"))
    self.apply_optimized_parameters(params)
```

---

## Storage Categories

### Quota Management

Each category has allocated GB quota to prevent one type from consuming all space:

| Category | Allocation | Max Size | Use Case |
|----------|-----------|----------|----------|
| **trading_state** | 0.5GB | 500MB | Daily snapshots, circuit breaker state |
| **options_greeks** | 1.5GB | 1.5GB | Historical Greeks, IV surfaces |
| **llm_cache** | 0.3GB | 300MB | Sentiment analysis cache |
| **backtest_results** | 0.5GB | 500MB | Optimization runs, metrics |
| **ml_models** | 1.5GB | 1.5GB | Trained models, feature sets |
| **monitoring_data** | 0.3GB | 300MB | Resource metrics, alerts |
| **research_data** | 0.4GB | 400MB | Analysis results, datasets |

### Category Enum

```python
from utils import StorageCategory

# Available categories
StorageCategory.TRADING_STATE      # Daily state snapshots
StorageCategory.OPTIONS_GREEKS     # Greeks history
StorageCategory.LLM_CACHE          # LLM results cache
StorageCategory.BACKTEST_RESULTS   # Backtest metrics
StorageCategory.ML_MODELS          # Trained models
StorageCategory.MONITORING_DATA    # System metrics
StorageCategory.RESEARCH_DATA      # Research datasets
```

---

## Monitoring & Alerts

### Automatic Monitoring

The algorithm monitors storage every 6 hours:

```python
# In algorithm Initialize()
# Automatically scheduled:
self.Schedule.On(
    self.DateRules.EveryDay(),
    self.TimeRules.Every(timedelta(hours=6)),
    self._check_storage,
)
```

### Alert Thresholds

| Threshold | Action | Response |
|-----------|--------|----------|
| **80% (4GB)** | Warning Alert | Log warning, suggest cleanup |
| **90% (4.5GB)** | Critical Alert | Trip circuit breaker, force cleanup |
| **100% (5GB)** | Storage Full | Halt trading, manual intervention |

### Viewing Storage Stats

#### In Algorithm Logs

```
Object Store: 2.35/5.00GB (47.0%)
Files: 12,543/50,000
```

#### In Final Report

```
==================================================
OBJECT STORE STATISTICS
==================================================
Storage Used: 2.35GB / 5.00GB (47.0%)
Files: 12,543 / 50,000
Growth Rate: 0.015GB/day
Days Until Full: ~177 days
Storage by Category:
  trading_state: 234.5MB (90 files)
  options_greeks: 1,123.2MB (8,432 files)
  llm_cache: 87.3MB (3,456 files)
  ...
```

### Manual Check

```python
# In algorithm
stats = self.storage_monitor.get_statistics()

self.Debug(f"Storage: {stats['current_usage_gb']:.2f}GB")
self.Debug(f"Growth: {stats['growth_rate_gb_per_day']:.3f}GB/day")

if stats['days_until_full'] and stats['days_until_full'] < 30:
    self.Debug(f"WARNING: Only {stats['days_until_full']} days until full")
```

---

## Best Practices

### 1. Use Compression for Large Objects

```python
# Automatic compression for objects >100KB
manager.save(
    key="large_dataset",
    data=large_data,
    category=StorageCategory.RESEARCH_DATA,
    # Automatically compressed if >100KB
)

# Force compression
manager.save(
    key="must_compress",
    data=data,
    category=StorageCategory.ML_MODELS,
    force_compress=True,
)
```

### 2. Set Expiration Dates

```python
# Trading state: 90 days
manager.save(..., expire_days=90)

# LLM cache: 30 days (news gets stale)
manager.save(..., expire_days=30)

# Backtest results: 1 year
manager.save(..., expire_days=365)

# ML models: no expiration
manager.save(..., expire_days=None)
```

### 3. Regular Cleanup

```python
# Automatic cleanup in algorithm
def _check_storage(self):
    if not self.storage_monitor.is_healthy():
        # Remove expired
        deleted = self.object_store_manager.cleanup_expired()
        self.Debug(f"Cleaned up {deleted} expired objects")

        # Check suggestions
        suggestions = self.storage_monitor.suggest_cleanup()
        for action in suggestions['actions']:
            self.Debug(f"Suggestion: {action['action']} ({action['reason']})")
```

### 4. Stay Under 45MB Per File

```python
# Check size before saving
import sys

json_str = json.dumps(data)
size_mb = sys.getsizeof(json_str) / (1024 * 1024)

if size_mb > 45:
    # Split into chunks or compress more
    self.Error(f"Object too large: {size_mb:.1f}MB")
else:
    manager.save(key, data, category)
```

### 5. Use Metadata for Searchability

```python
manager.save(
    key="backtest_2025_01_15",
    data=results,
    category=StorageCategory.BACKTEST_RESULTS,
    expire_days=365,
    metadata={
        "strategy": "two_part_spread",
        "sharpe": 1.8,
        "max_drawdown": 0.12,
        "version": "v2.3",
    }
)

# Later: Search by metadata
obj = manager.get_metadata("backtest_2025_01_15")
if obj and obj.metadata.get("sharpe", 0) > 1.5:
    # Load this backtest
    results = manager.load("backtest_2025_01_15")
```

---

## Troubleshooting

### Storage Full

**Symptoms:**
- "Storage critical: 90%+ full" alerts
- Circuit breaker trips
- Save operations fail

**Solutions:**

1. **Check usage:**
   ```python
   stats = self.storage_monitor.get_statistics()
   for category, stats in stats['by_category'].items():
       print(f"{category}: {stats['size_mb']:.1f}MB")
   ```

2. **Remove expired:**
   ```python
   deleted = self.object_store_manager.cleanup_expired()
   ```

3. **Manually clean category:**
   ```python
   # Get all keys in category
   keys = self.object_store_manager.list_keys(StorageCategory.LLM_CACHE)

   # Delete old cache entries
   for key in keys[:100]:  # Delete oldest 100
       self.object_store_manager.delete(key)
   ```

4. **Upgrade tier:**
   - Consider 10GB tier ($50/month) if consistently over 80%

### File Too Large Error

**Error:** "Object too large: 52.3MB > 45MB"

**Solutions:**

1. **Enable compression:**
   ```python
   manager.save(..., force_compress=True)
   ```

2. **Split into chunks:**
   ```python
   # Split large dataset
   chunk_size = 1000
   for i, chunk in enumerate(chunks(data, chunk_size)):
       manager.save(
           key=f"data_chunk_{i}",
           data=chunk,
           category=StorageCategory.RESEARCH_DATA,
       )
   ```

3. **Store reference instead:**
   ```python
   # Instead of storing full dataset, store summary
   summary = {
       "location": "s3://mybucket/data.csv",
       "hash": hashlib.md5(data).hexdigest(),
       "rows": len(data),
   }
   manager.save(key, summary, category)
   ```

### Slow Save/Load

**Symptoms:**
- Save operations take >5 seconds
- Load operations timeout

**Solutions:**

1. **Reduce object size:**
   ```python
   # Only save essential data
   state = {
       "positions": {k: v for k, v in positions.items() if v['quantity'] != 0},
       "key_metrics": {
           "equity": equity,
           "daily_pnl": daily_pnl,
       }
   }
   ```

2. **Use compression:**
   - Reduces network transfer time
   - Reduces storage I/O

3. **Batch operations:**
   ```python
   # Instead of 100 individual saves
   # Save 1 aggregated object
   daily_greeks = {
       symbol: greeks_data
       for symbol, greeks_data in all_greeks.items()
   }
   manager.save("greeks_daily", daily_greeks, ...)
   ```

### Cannot List Keys in Live Trading

**Issue:** ObjectStore.GetEnumerator() not available in live trading

**Solution:** Track keys in algorithm:

```python
class OptionsTradingBot(QCAlgorithm):
    def Initialize(self):
        # Track saved keys
        self._saved_keys = set()

    def save_data(self, key, data):
        self.object_store_manager.save(key, data, ...)
        self._saved_keys.add(key)

    def list_my_keys(self):
        return list(self._saved_keys)
```

---

## Cost Summary

### Monthly Cost Breakdown

```
QuantConnect Services:
  Compute Nodes (B8-16 + R8-16 + L2-4):  $92/month
  Object Store (5GB):                    $20/month
  ──────────────────────────────────────────────
  TOTAL:                                 $112/month
```

### Storage ROI

**Value Provided:**
- ✅ **Continuous operation:** State persists across restarts
- ✅ **Faster development:** Cache expensive LLM calls
- ✅ **Better strategies:** Archive and analyze backtest results
- ✅ **Research → Live:** Seamless parameter transfer
- ✅ **Risk management:** Historical circuit breaker data
- ✅ **Compliance:** Audit trail of all trading decisions

**Cost per GB:** $4/GB/month
**Cost per file:** $0.0004/file/month

---

## Migration Guide

### From 2GB Tier

If upgrading from 2GB:

1. **No code changes needed** - same API
2. **Update config:**
   ```json
   {
     "object_store": {
       "tier": "5GB",
       "max_storage_gb": 5,
       "max_files": 50000
     }
   }
   ```
3. **Adjust allocations** if needed

### From 50MB Free Tier

1. **Enable Object Store:**
   ```json
   {
     "object_store": {
       "enabled": true,
       "tier": "5GB"
     }
   }
   ```

2. **Add to algorithm:**
   ```python
   if self.object_store_manager:
       # Now safe to use
       self.object_store_manager.save(...)
   ```

---

## Additional Resources

- **Object Store API:** [utils/object_store.py](../../utils/object_store.py)
- **Storage Monitor:** [utils/storage_monitor.py](../../utils/storage_monitor.py)
- **Management Script:** [scripts/manage_object_store.py](../../scripts/manage_object_store.py)
- **QuantConnect Docs:** [Object Store Documentation](https://www.quantconnect.com/docs/v2/cloud-platform/organizations/object-store)
- **Pricing:** [QuantConnect Pricing](https://www.quantconnect.com/pricing/)

---

**Last Updated:** 2025-11-30
**Version:** 1.0.0
