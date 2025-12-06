# QuantConnect Compute Nodes Guide

Complete guide to selecting and optimizing QuantConnect compute nodes for the options trading bot.

## Table of Contents

- [Node Overview](#node-overview)
- [Recommended Configuration](#recommended-configuration)
- [Node Selection Strategy](#node-selection-strategy)
- [Resource Monitoring](#resource-monitoring)
- [Deployment Guide](#deployment-guide)
- [Cost Optimization](#cost-optimization)
- [Troubleshooting](#troubleshooting)

## Node Overview

QuantConnect offers three types of compute nodes for different stages of algorithm development:

### Backtesting Nodes

| Model | Cores | Speed (GHz) | RAM (GB) | GPU | Cost/Month | Best For |
|-------|-------|-------------|----------|-----|------------|----------|
| **B-MICRO** | 2 | 3.3 | 8 | No | $0 (20s delay) | Simple strategies, testing |
| **B2-8** | 2 | 4.9 | 8 | No | $14 | Basic strategies |
| **B4-12** | 4 | 4.9 | 12 | No | $20 | Moderate complexity |
| **B8-16** | 8 | 4.9 | 16 | No | $28 | **Options trading (RECOMMENDED)** |
| **B4-16-GPU** | 4 | 3.0 | 16 | Yes | $400 | Local ML training |

### Research Nodes

| Model | Cores | Speed (GHz) | RAM (GB) | GPU | Cost/Month | Best For |
|-------|-------|-------------|----------|-----|------------|----------|
| **R1-4** | 1 | 2.4 | 4 | No | $5 | Basic notebooks |
| **R2-8** | 2 | 2.4 | 8 | No | $10 | Light research |
| **R4-12** | 4 | 2.4 | 12 | No | $14 | Moderate research |
| **R8-16** | 8 | 2.4 | 16 | No | $14 | **LLM ensemble (RECOMMENDED)** |
| **R4-16-GPU** | 4 | 3.0 | 16 | Yes | $400 | Local ML training |

### Live Trading Nodes

| Model | Cores | Speed (GHz) | RAM (GB) | GPU | Cost/Month | Colocated | Max Latency |
|-------|-------|-------------|----------|-----|------------|-----------|-------------|
| **L-MICRO** | 1 | 2.6 | 0.5 | No | $20 | Yes | <100ms | Too small for options |
| **L1-1** | 1 | 2.6 | 1 | No | $25 | Yes | <100ms | Basic strategies |
| **L1-2** | 1 | 2.6 | 2 | No | $35 | Yes | <100ms | Small options portfolios |
| **L2-4** | 2 | 2.6 | 4 | No | $50 | Yes | <100ms | **Options trading (RECOMMENDED)** |
| **L8-16-GPU** | 8 | 3.1 | 16 | Yes | $400 | Yes | <100ms | Real-time ML inference |

## Recommended Configuration

For this options trading bot with LLM integration:

```json
{
  "backtesting": "B8-16",
  "research": "R8-16",
  "live_trading": "L2-4",
  "object_store": "5GB",
  "total_cost": "$112/month"
}
```

### Complete Infrastructure Cost

```
Backtesting (B8-16):     $28/month
Research (R8-16):        $14/month
Live Trading (L2-4):     $50/month
Object Store (5GB):      $20/month
────────────────────────────────────
TOTAL:                   $112/month
```

### Why These Nodes?

#### B8-16 for Backtesting

- **8 cores @ 4.9 GHz**: Fast parallel processing for options Greeks calculations
- **16GB RAM**: Handles 500+ option contracts simultaneously
- **16GB RAM**: Multi-chain spread analysis without memory pressure
- **Cost-effective**: Best performance per dollar for options strategies

**Memory breakdown:**
```
Base overhead:        1,100 MB
Securities (10):        50 MB
Option contracts (500): 2,500 MB
Greeks calculations:    200 MB
Scanners:              150 MB
Risk management:       100 MB
Two-part spread:       200 MB
LLM API calls:          50 MB
-----------------------------------
Total:                 4,350 MB (4.3 GB)
Safety margin:        11.7 GB remaining
```

#### R8-16 for Research

- **8 cores @ 2.4 GHz**: Sufficient for LLM API calls (not local inference)
- **16GB RAM**: Jupyter notebooks with full data analysis
- **Best value**: Same RAM as R4-16-GPU but $14 vs $400/month
- **No GPU needed**: LLM providers (GPT-4o, Claude) run on their servers

**Use cases:**
- Strategy exploration in notebooks
- Volatility surface modeling
- Monte Carlo simulations
- Walk-forward optimization
- Backtesting parameter sweeps

#### L2-4 for Live Trading

- **2 cores @ 2.6 GHz**: Core 1 for trading loop, Core 2 for scanners
- **4GB RAM**: Comfortable headroom for 500 contracts
- **Sub-100ms latency**: NY7 colocation near Charles Schwab
- **Critical for execution**: 2.5-second cancel logic needs CPU responsiveness

**Why not L1-2?**
- Options chains expand (new strikes, expirations)
- No safety margin for market volatility
- Risk of memory crashes during high-volume periods

**Why not L8-16-GPU?**
- LLM calls are API-based (external processing)
- GPU won't accelerate HTTP requests
- Save $350/month ($400 - $50)

## Node Selection Strategy

### Automated Selection

Use the `deploy_with_nodes.py` script for automatic node recommendation:

```bash
# Analyze algorithm and get recommendations
python scripts/deploy_with_nodes.py algorithms/options_trading_bot.py --analyze-only

# Output:
# ============================================================
# QUANTCONNECT COMPUTE NODE RECOMMENDATIONS
# ============================================================
#
# BACKTESTING:
#   Recommended: B8-16
#   Cores: 8 @ 4.9GHz
#   RAM: 16GB
#   Cost: $28/month
#   Reason: High memory requirements (4.3GB) or options trading
#   Estimated Memory: 4.35GB
#
# RESEARCH:
#   Recommended: R8-16
#   Cores: 8 @ 2.4GHz
#   RAM: 16GB
#   Cost: $14/month
#   Reason: High memory or LLM ensemble analysis (4.3GB)
#   Estimated Memory: 4.35GB
#
# LIVE_TRADING:
#   Recommended: L2-4
#   Cores: 2 @ 2.6GHz
#   RAM: 4GB
#   Cost: $50/month
#   Reason: Options trading with multiple chains (4.3GB)
#   Estimated Memory: 4.35GB
#
# TOTAL MONTHLY COST: $92/month
# ============================================================
```

### Manual Selection

Use the `NodeOptimizer` class directly:

```python
from utils import NodeOptimizer, AlgorithmRequirements

# Define your requirements
requirements = AlgorithmRequirements(
    num_securities=10,
    num_option_chains=5,
    contracts_per_chain=100,
    max_concurrent_positions=20,
    use_llm_analysis=True,
    use_ml_training=False,  # API-based, not local
)

# Get recommendations
optimizer = NodeOptimizer()
recommendations = optimizer.recommend_nodes(requirements)

# Access specific recommendations
backtest_node = recommendations["backtesting"]["node"].model
print(f"Use {backtest_node} for backtesting")
```

## Resource Monitoring

### Real-Time Monitoring

The `ResourceMonitor` class tracks system resources during live trading:

```python
from utils import create_resource_monitor

# Create monitor with thresholds
monitor = create_resource_monitor(
    config={
        "memory_warning_pct": 80,
        "memory_critical_pct": 90,
        "cpu_warning_pct": 75,
        "cpu_critical_pct": 85,
    },
    circuit_breaker=circuit_breaker,
)

# In trading loop (every 30 seconds)
metrics = monitor.update(
    active_securities=len(securities),
    active_positions=len(positions),
    broker_latency_ms=latency,
)

# Check health
if not monitor.is_healthy():
    # Reduce load or halt trading
    print("WARNING: System resources under pressure")
```

### Memory Estimation

Estimate memory before adding securities:

```python
# Can we add 5 more option chains?
can_add = monitor.can_add_securities(
    num_securities=500,  # 5 chains × 100 contracts
    node_ram_gb=4,       # L2-4 node
)

if not can_add:
    print("Insufficient memory for additional chains")
```

### Statistics and Alerts

```python
# Get resource usage statistics
stats = monitor.get_statistics()
print(f"Memory: {stats['memory']['avg_pct']:.1f}% average")
print(f"CPU: {stats['cpu']['avg_pct']:.1f}% average")
print(f"Alerts: {stats['alerts']['total']} total")

# Get recent alerts
alerts = monitor.get_recent_alerts(limit=5)
for alert in alerts:
    print(f"{alert.severity}: {alert.message}")
```

### Automatic Integration

The `OptionsTradingBot` algorithm includes automatic monitoring:

```python
# Monitoring runs every 30 seconds (configurable)
# config/settings.json:
{
  "quantconnect": {
    "monitoring": {
      "enabled": true,
      "check_interval_seconds": 30,
      "log_metrics": true,
      "alert_on_high_usage": true
    }
  }
}
```

## Deployment Guide

### 1. Local Analysis

Before deploying, analyze your algorithm:

```bash
# Check resource requirements
python scripts/deploy_with_nodes.py algorithms/options_trading_bot.py --analyze-only
```

### 2. Backtest Deployment

Deploy to backtesting node:

```bash
# Automatic node selection
python scripts/deploy_with_nodes.py \
    algorithms/options_trading_bot.py \
    --type backtest

# Manual node override
python scripts/deploy_with_nodes.py \
    algorithms/options_trading_bot.py \
    --type backtest \
    --node B8-16
```

### 3. Research Deployment

Launch research environment:

```bash
python scripts/deploy_with_nodes.py \
    algorithms/options_trading_bot.py \
    --type research \
    --node R8-16
```

### 4. Live Trading Deployment

Deploy to live trading (requires approval):

```bash
python scripts/deploy_with_nodes.py \
    algorithms/options_trading_bot.py \
    --type live \
    --node L2-4 \
    --project-id YOUR_PROJECT_ID
```

### 5. Monitoring Deployment

After deployment, monitor resources:

```bash
# View metrics log
tail -f logs/resource_metrics.json

# Check for alerts
grep "CRITICAL" logs/trading_bot.log
```

## Cost Optimization

### Current Configuration Cost

```
Backtesting (B8-16):     $28/month
Research (R8-16):        $14/month
Live Trading (L2-4):     $50/month
──────────────────────────────────
Total:                   $92/month
```

### Scaling Strategies

#### Phase 1: Development ($38/month)

Start with smaller nodes during development:

```
Backtesting (B4-12):     $20/month
Research (R4-12):        $14/month
Live Trading (paper):    $0/month
──────────────────────────────────
Total:                   $34/month
```

Upgrade to B8-16 when you hit memory limits.

#### Phase 2: Production ($92/month)

Use recommended configuration for live trading.

#### Phase 3: Scale ($142/month)

Add second backtest node for parallel testing:

```
Backtesting 1 (B8-16):   $28/month
Backtesting 2 (B8-16):   $28/month
Research (R8-16):        $14/month
Live Trading (L2-4):     $50/month
──────────────────────────────────
Total:                   $120/month
```

### When NOT to Use GPU Nodes

**Don't use GPU ($400/month) if:**
- Using LLM APIs (GPT-4o, Claude) ✗
- Using pre-trained models (FinBERT via API) ✗
- Standard indicators (RSI, MACD) ✗
- Greeks calculations (handled by QC) ✗

**Use GPU only if:**
- Training custom ML models locally ✓
- Running local FinBERT inference (not API) ✓
- Real-time neural network predictions ✓
- Reinforcement learning for execution ✓

### Cost vs Performance Matrix

| Node | Cost/Month | Speed | RAM | Use When |
|------|------------|-------|-----|----------|
| B-MICRO | $0 | Low | 8GB | Testing only (20s delay) |
| B4-12 | $20 | High | 12GB | Small options portfolios |
| **B8-16** | **$28** | **Very High** | **16GB** | **Options trading (BEST VALUE)** |
| B4-16-GPU | $400 | Medium | 16GB | Local ML training only |

## Troubleshooting

### Memory Exhaustion

**Symptoms:**
- Algorithm crashes during live trading
- "Out of memory" errors in logs
- Resource monitor shows >90% memory usage

**Solutions:**
1. Reduce number of option chains:
   ```python
   # In config/settings.json
   "resource_limits": {
       "max_option_chains": 3  // Reduce from 5
   }
   ```

2. Upgrade to larger node:
   ```bash
   # L2-4 (4GB) → L8-16-GPU (16GB)
   python scripts/deploy_with_nodes.py \
       algorithms/options_trading_bot.py \
       --type live \
       --node L8-16-GPU
   ```

3. Reduce contracts per chain:
   ```python
   # In options_scanner config
   "max_days_to_expiry": 30  // Reduce from 45
   ```

### High CPU Usage

**Symptoms:**
- CPU consistently >85%
- Slow order execution
- Missed profit-taking opportunities

**Solutions:**
1. Reduce scan frequency:
   ```python
   "movement_scanner": {
       "scan_interval_seconds": 120  // Increase from 60
   }
   ```

2. Disable non-critical features:
   ```python
   "technical_indicators": {
       "ichimoku": {"enabled": false}  // Disable heavy indicators
   }
   ```

3. Upgrade to more cores (L2-4 → L8-16-GPU)

### High Latency

**Symptoms:**
- Broker latency >100ms
- Slow order fills
- Two-part spreads timing out

**Solutions:**
1. Verify node colocation:
   ```python
   # All L-series nodes are colocated in NY7
   # Check with QuantConnect support if latency persists
   ```

2. Reduce order complexity:
   ```python
   "order_execution": {
       "cancel_after_seconds": 2.5  // Quick cancel
   }
   ```

3. Check network issues with QC support

### Resource Alerts

**Symptoms:**
- Frequent "memory_warning" alerts
- Circuit breaker trips due to resources

**Solutions:**
1. Review resource statistics:
   ```python
   # At end of algorithm
   stats = resource_monitor.get_statistics()
   # Check max memory usage
   if stats['memory']['max_pct'] > 85:
       # Upgrade node or reduce load
   ```

2. Implement load shedding:
   ```python
   # In algorithm
   if not resource_monitor.is_healthy():
       # Close least profitable positions
       # Stop opening new positions
       # Reduce scan frequency
   ```

## Best Practices

### 1. Always Monitor Resources

Enable monitoring in production:

```json
{
  "quantconnect": {
    "monitoring": {
      "enabled": true,
      "check_interval_seconds": 30,
      "alert_on_high_usage": true
    }
  }
}
```

### 2. Set Realistic Limits

Configure conservative limits:

```json
{
  "resource_limits": {
    "max_option_chains": 5,
    "max_contracts_per_chain": 100,
    "memory_warning_pct": 80,
    "memory_critical_pct": 90
  }
}
```

### 3. Use Circuit Breaker Integration

Link resource monitor to circuit breaker:

```python
monitor = create_resource_monitor(
    config=config,
    circuit_breaker=circuit_breaker,  # Auto-halt on critical
)
```

### 4. Analyze Before Deploying

Always run analysis first:

```bash
# Check requirements
python scripts/deploy_with_nodes.py \
    algorithms/my_strategy.py \
    --analyze-only

# Review recommendations before deploying
```

### 5. Start Small, Scale Up

Begin with smaller nodes in paper trading:

```
Paper Trading: L1-2 ($35/month)
↓
Small Live:    L2-4 ($50/month)
↓
Full Scale:    L2-4 + multiple strategies
```

### 6. Monitor Cost vs Performance

Track monthly costs and adjust:

```python
# Monthly cost tracking
costs = {
    "backtesting": 28,
    "research": 14,
    "live_trading": 50,
}

# If profitability justifies it:
# - Add second live node for redundancy
# - Add GPU node for ML features
# - Scale to larger option universe
```

## Object Store Integration

The 5GB Object Store tier complements the compute nodes for complete infrastructure:

- **Trading State:** Persist positions and circuit breaker status
- **Greeks History:** Store 180 days of options data
- **LLM Cache:** Cache sentiment analysis results
- **Backtest Results:** Archive optimization runs
- **ML Models:** Deploy trained models from research

See [Object Store Guide](OBJECT_STORE.md) for details.

## Additional Resources

- [QuantConnect Documentation](https://www.quantconnect.com/docs/v2/cloud-platform/organizations/resources)
- [Node Optimizer Source](../../utils/node_optimizer.py)
- [Resource Monitor Source](../../utils/resource_monitor.py)
- [Object Store Guide](OBJECT_STORE.md)
- [Deployment Script](../../scripts/deploy_with_nodes.py)
- [Configuration Reference](../../config/settings.json)

---

**Last Updated:** 2025-11-30
**Version:** 1.1.0
