# QuantConnect Data Subscription Recommendations

Analysis of dataset subscriptions for the options trading bot with LLM integration and two-part spread strategy.

**Last Updated:** 2025-11-30

---

## What You Already Have (FREE)

### From QuantConnect (Included)

✅ **US Equity Options** (AlgoSeek - Free in Cloud)
- Minute-level pricing data
- Strike prices and expirations
- Open interest
- Greeks (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility
- Coverage: 4,000 symbols since 2012

✅ **US Equities** (Free)
- Trade and quote data
- Daily bars
- Splits and dividends
- Symbol changes

✅ **US Equity Option Universe** (Free)
- Available contracts
- Daily Greeks
- Daily IV values

### From Charles Schwab API (Free via Broker)

✅ **Real-Time Market Data**
- Live quotes for your positions
- Full option chains with Greeks
- Implied volatility
- Real-time execution data

**Total Current Cost:** $0/month for data

---

## Recommended Subscriptions

### Tier 1: Essential ($385/year total)

These provide immediate value for your strategy:

#### 1. **Quiver Quantitative Bundle** ($330/year)

| Dataset | Cost | Value Proposition |
|---------|------|-------------------|
| **Congress Trading** | $55/yr | Track senator/rep trades for directional signals |
| **Insider Trading** | $55/yr | Corporate insider activity (2-day disclosure) |
| **WallStreetBets** | $55/yr | Retail sentiment for meme stock detection |
| **CNBC Trading** | $55/yr | Media personality trades |
| **Government Contracts** | $55/yr | Federal contract awards (bullish signal) |
| **Corporate Lobbying** | $55/yr | Lobbying spending (regulatory insight) |

**Why This Matters:**
- ✅ **News Corroboration:** Your movement scanner requires news validation - insider/congress activity is highly correlated with price movements
- ✅ **Early Signals:** Congress trades disclosed within 45 days, insiders within 2 days
- ✅ **LLM Enhancement:** Feed this data to your sentiment ensemble for better predictions
- ✅ **Two-Part Spread Timing:** Know when smart money is positioning before major moves

**Integration Example:**
```python
# Check if Congress/insiders are buying before opening spread
congress_trades = self.CongressTrading["AAPL"]
insider_trades = self.InsiderTrading["AAPL"]

if congress_trades.BuyTransactions > 3 and insider_trades.NetBuying > 0:
    # Bullish signal - execute two-part spread
    self.execute_spread(symbol, bullish=True)
```

#### 2. **US Equity Option Universe Updates** ($1,000/year)

**Why:**
- ✅ **Fresh Greeks:** Daily updates to Greeks and IV
- ✅ **New Contracts:** Catch weekly options as they're listed
- ✅ **Accurate Pricing:** Your underpriced scanner needs current IV
- ✅ **Production Ready:** Critical for live trading

**Without This:**
- ❌ Stale Greeks from historical data
- ❌ Miss newly listed contracts
- ❌ Incorrect IV calculations
- ❌ Poor spread pricing

**ROI Calculation:**
- Cost: $1,000/year ($83/month)
- Value: Avoid 1 bad trade/month = saves >$100
- **Break-even:** 1 avoided bad trade

### Tier 2: High Value ($605/year additional)

#### 3. **Brain Social Sentiment Bundle** ($550/year)

| Dataset | Cost | Coverage |
|---------|------|----------|
| **Social Sentiment** | $275/yr | 4,500 stocks, 7-day & 30-day scores |
| **Social Sentiment Universe** | $275/yr | Full universe updates |

**Why:**
- ✅ **LLM Validation:** Cross-check your FinBERT/GPT-4o/Claude ensemble
- ✅ **2,000+ Sources:** Monitors financial media in 33 languages
- ✅ **Daily Updates:** Sentiment scores -1 to +1
- ✅ **Complement LLMs:** Cheaper than API calls at scale

**vs Your Current LLM Setup:**
- LLM APIs: ~$0.002/analysis × 1,000/day = $60/month ($720/year)
- Brain Sentiment: $550/year (flat rate)
- **Savings:** $170/year + faster backtests

**Integration:**
```python
# Ensemble: Your LLMs + Brain sentiment
llm_sentiment = self.llm_ensemble.analyze(headline)
brain_sentiment = self.BrainSentiment["AAPL"].SevenDaySentiment

# Weighted average
final_sentiment = (llm_sentiment * 0.6) + (brain_sentiment * 0.4)
```

#### 4. **US Equity Coarse Universe Updates** ($200/year)

**Why:**
- ✅ **Fresh Fundamentals:** Daily updates to market cap, volume, price
- ✅ **Scanner Fuel:** Your movement scanner needs current data
- ✅ **Symbol Changes:** Track ticker changes, delistings
- ✅ **Production Critical:** Stale data = missed opportunities

**Without This:**
- ❌ Miss hot stocks (new IPOs, momentum)
- ❌ Trade delisted symbols
- ❌ Incorrect universe filtering

---

## Additional Datasets Analyzed (Not Recommended for Options Trading)

### ❌ Quiver Wikipedia Page Views ($165/year)

**What It Is:**
- Tracks Wikipedia page views for 1,300 US Equities since October 2016
- Monitors volatility in web traffic as a proxy for price volatility
- Includes week-over-week and month-over-month percent changes

**Why Skip for Options Trading:**
- More relevant for meme stock/retail sentiment (already covered by WSB dataset for $55/yr)
- Wikipedia traffic is a lagging indicator, not predictive for options pricing
- Your movement scanner + news corroboration is more timely
- **Alternative:** WSB dataset at $55/yr provides better retail sentiment at 1/3 the cost

**Sources:** [QuantConnect Wikipedia Views](https://www.quantconnect.com/data/quiver-quantitative-wikipedia-views), [Quiver Quantitative](https://www.quiverquant.com/)

---

### ❌ RegAlytics US Regulatory Alerts Feed ($110/year)

**What It Is:**
- Tracks changes from 8,000+ global governing bodies
- 2.5 million alerts since January 2020, delivered daily by 8AM
- Monitors regulatory phrases like "emergency rule" or "proposed rule change"

**Why Skip for Options Trading:**
- More relevant for regulatory compliance and sector-wide risk management
- Regulatory changes affect broad sectors, not individual options pricing
- Your options strategy focuses on Greeks/IV, not regulatory risk
- Better suited for long-term fundamental strategies

**Sources:** [RegAlytics QuantConnect](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/regalytics), [RegAlytics.ai](https://www.regalytics.ai)

---

### ❌ ExtractAlpha Estimize ($825/year)

**What It Is:**
- Crowdsourced EPS/revenue estimates from 120,000+ contributors
- Covers 2,800+ US Equities since January 2011
- 70% more accurate than sell-side estimates
- Enables post-earnings announcement drift (PEAD) strategies

**Why Skip for Options Trading:**
- **Earnings-focused, not options Greeks-focused**
- Your strategy trades on IV/Delta/Greeks, not earnings surprises
- Earnings plays are binary events (high risk for spreads)
- Better for fundamental equity strategies than technical options trading

**If You Were Trading Earnings:**
- Strong dataset for earnings plays
- Could combine with options volatility around earnings
- But your current strategy avoids earnings events

**Sources:** [ExtractAlpha Estimize](https://extractalpha.com/solutions/), [QuantConnect Estimize](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/extractalpha/estimize)

---

### ❌ ExtractAlpha True Beats ($825/year)

**What It Is:**
- EPS and revenue predictions for US-listed equities
- Used for earnings surprise arbitrage and sentiment trading
- Enables stock/sector selection based on fundamental predictions

**Why Skip:**
- Same reasoning as Estimize - earnings-focused, not options Greeks
- Your bot doesn't trade earnings events
- $825/yr for earnings data you won't use

**Sources:** [ExtractAlpha True Beats](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/extractalpha/true-beats)

---

### ❌ ExtractAlpha Tactical Model ($825/year)

**What It Is:**
- Stock scoring algorithm capturing technical dynamics over 1-10 day horizons
- 37.7% annualized outperformance with 2.85 Sharpe ratio in backtests
- Combines momentum and seasonality for short-term equity trades

**Why Skip:**
- **Equity scoring model, not options-specific**
- Your options strategy already uses technical indicators (RSI, MACD, Ichimoku)
- Designed for directional equity trading, not spread construction
- You'd need to translate equity scores to options strategies manually
- $825/yr for data you'd have to heavily adapt

**If You Were Trading Equities:**
- Strong alpha signal for short-term equity rotation
- But you're focused on options spreads, not stock picking

**Sources:** [ExtractAlpha Tactical](https://extractalpha.com/fact-sheet/tactical-model/), [QuantConnect Tactical](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/extractalpha/tactical)

---

### ❌ ExtractAlpha Cross Asset Model ($825/year)

**What It Is:**
- Stock scoring based on options market trading activity
- Uses put-call spread, volatility skewness, and volume
- 2.46 Sharpe ratio in backtests
- Leverages options data to predict equity price trends

**Why Skip:**
- Designed for **equity prediction using options data**, not options trading itself
- You already have direct access to options data (Greeks, IV, volume, OI)
- Using options data to trade equities is backwards for your use case
- $825/yr for indirect signal when you have direct data

**What's Interesting:**
- Uses options market sentiment to predict equities
- Your strategy does the opposite: uses equity/news to trade options

**Sources:** [ExtractAlpha Cross Asset](https://extractalpha.com/fact-sheet/cross-asset-model/), [QuantConnect Cross Asset](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/extractalpha/cross-asset-model)

---

### ❌ Brain ML Stock Ranking ($275/year + $275/year Universe)

**What It Is:**
- ML-generated daily rankings for 1,000 largest US stocks
- Predicts future returns across 2, 3, 5, 10, and 21-day horizons
- Score ranges from -1 to +1 (confidence in quintile placement)
- Ensemble of ML classifiers with features: fundamentals, price-volume, volatility, calendar anomalies

**Why Skip:**
- **Stock ranking system, not options signals**
- Your options scanner already identifies underpriced contracts via Greeks/IV
- ML stock predictions don't directly translate to options spread opportunities
- Would need additional work to convert stock rankings to options strategies
- **Total cost:** $550/year for equity signals you'd have to adapt

**If Useful:**
- Could use top-ranked stocks as universe filter for options scanning
- But your movement scanner already identifies good underlying candidates

**Sources:** [Brain ML Ranking](https://braincompany.co/bsr.html), [QuantConnect Brain ML](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/brain/brain-ml-stock-ranking)

---

### ❌ Brain Company Filings Analysis ($275/year + $275/year Universe)

**What It Is:**
- NLP analysis of SEC filings (10-K, 10-Q) for 5,000+ US equities since 2010
- Metrics: financial sentiment, language types (litigious), readability, document similarity
- Analyzes specific sections (Risk Factors, MD&A)
- Updated daily with largest updates around earnings season

**Why Skip:**
- **Fundamental analysis for long-term investors, not options traders**
- SEC filings are quarterly/annual - too slow for options strategies
- Your options trades are 7-45 DTE, filings analysis is for months/years
- Already have faster sentiment via LLM ensemble and Brain Social Sentiment
- **Total cost:** $550/year for slow-moving fundamental data

**Sources:** [Brain Filings Analysis](https://braincompany.co/blmcf.html), [QuantConnect Brain Filings](https://www.quantconnect.com/data/brain-language-metrics-company-filings)

---

### ❌ Smart Insider Buyback Intentions ($120/year)

**What It Is:**
- Tracks corporate share buyback announcements and authorizations
- Monitors stated intentions vs. executed trades to predict future buybacks
- Compares buyback patterns to establish predictive models

**Why Skip:**
- Buyback announcements are **long-term bullish signals** (months to execute)
- Your options strategy trades 7-45 DTE (too short for buyback impact)
- Buybacks don't directly affect options Greeks or IV
- More relevant for fundamental equity investors

**If Useful:**
- Could identify long-term bullish underlying stocks
- But your strategy focuses on short-term options mispricing, not fundamental longs

**Sources:** [Smart Insider](https://www.smartinsider.com/share-buybacks/), [Smart Insider Buybacks](https://www.smartinsider.com/buybacks/)

---

### ❌ Smart Insider Buyback Transactions ($120/year)

**What It Is:**
- Actual executed buyback trades (not just intentions)
- Real-time tracking of corporate repurchase activity

**Why Skip:**
- Same reasoning as Buyback Intentions
- Actual buybacks are even more long-term (program execution over months)
- Doesn't impact short-term options pricing or Greeks

**Sources:** [Smart Insider](https://www.smartinsider.com/)

---

### ❌ Kavout Composite Factor Bundle ($429/year)

**What It Is:**
- Ensemble ML scores for 5 systematic factors: Quality, Value, Momentum, Growth, Low Volatility
- Daily scores for all US stocks since 2003
- Each signal combines hundreds of anomalies
- Adopted by multi-billion dollar quant funds

**Why Skip:**
- **Factor investing framework for equities, not options**
- These factors (value, quality, growth) are for stock selection
- Your options strategy doesn't select stocks - it finds mispriced options
- Movement scanner + technical indicators already capture momentum
- $429/yr for equity factor signals you won't use for options spreads

**If You Were Running Long-Only Equity:**
- Excellent dataset for systematic factor investing
- But you're trading options on already-selected underlyings

**Sources:** [Kavout Composite Bundle](https://www.quantconnect.com/data/kavout-composite-factor-bundle), [Kavout Factor Investing](https://www.kavout.com/factor-investing/)

---

### ❌ US Equity Security Master ($500/year one-time)

**What It Is:**
- Tracks corporate actions: splits, dividends, delistings, mergers, ticker changes
- Covers 27,500 US Equities since January 1998
- Automatically handled by LEAN engine

**Why Skip:**
- **Already included FREE with QuantConnect cloud**
- LEAN automatically loads this when you request US Equities data
- No additional purchase needed - it's built into the platform
- Corporate actions are automatically adjusted in your data

**Sources:** [QuantConnect Security Master](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/quantconnect/us-equity-security-master)

---

### ❌ US ETF Constituents ($3,000 history + $1,000/year updates)

**What It Is:**
- Tracks constituents and weighting of 2,650 US ETFs
- Daily updates since 2009 (monthly before 2015)
- Enables basket trading and ETF rebalancing strategies

**Why Skip:**
- **ETF basket trading, not options strategies**
- Your bot trades individual equity options, not ETF constituent baskets
- Would need if building ETF arbitrage or constituent rotation strategies
- Not relevant for underpriced options scanner or two-part spreads

**If You Were Trading ETF Arbitrage:**
- Strong dataset for ETF/basket strategies
- But you're focused on single-name options

**Sources:** [QuantConnect ETF Constituents](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/quantconnect/us-etf-constituents)

---

## Not Recommended (Poor ROI)

### ❌ US Equity Option Universe History ($3,300)

**Why Skip:**
- You already have FREE historical Greeks in QC cloud
- Only needed if downloading for local backtesting
- Your bot runs in QC cloud, not locally

### ❌ AlgoSeek Minute/Tick Data ($600-16,800/year)

**Why Skip:**
- You have FREE minute data in QC cloud
- Your strategy uses 2.5-second cancel logic (minute data sufficient)
- Tick data only needed for HFT (not your use case)

### ❌ Benzinga News Feed ($1,440/year)

**Why Skip:**
- Expensive for news you can get via APIs
- Your LLM ensemble already analyzes news
- Quiver datasets provide better actionable signals

### ❌ Benzinga News Feed Download ($600/year one-time)

**Why Skip:**
- Same as Benzinga News Feed but for local download
- You run in QuantConnect cloud, not locally
- Already have LLM news analysis

---

## Other Asset Classes (Not Applicable)

These datasets cover asset classes outside your equity options focus:

### ❌ US Futures Datasets ($500-$1,000 one-time + $800-$1,200/year)

**What It Is:**
- Futures Security Master: Mapping data for 162 CME futures contracts (S&P E-mini, etc.)
- US Futures data by AlgoSeek in various resolutions
- Supports continuous contract construction with rolling techniques

**Why Skip:**
- **You're trading equity options, not futures**
- Different market mechanics, margin requirements, contract specifications
- No overlap with your options Greeks/spread strategies

**Sources:** [QuantConnect US Futures](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/quantconnect/us-futures-security-master)

---

### ❌ US Index Options ($1,200 history + $960/year updates)

**What It Is:**
- Options on SPX (S&P 500), VIX (Volatility Index), NDX (Nasdaq 100)
- European-style index options (cash-settled)
- Includes Greeks, IV, minute resolution since January 2012
- FREE on QuantConnect Cloud

**Why Skip (Maybe):**
- **Different from equity options** - European-style, cash-settled, different Greeks behavior
- VIX options have unique volatility-of-volatility dynamics
- SPX options use AM/PM settlement (more complex)
- Your strategy is optimized for American-style equity options

**If You Want Index Options:**
- Good news: **Already FREE in QuantConnect Cloud**
- No subscription needed for SPX/VIX/NDX options data
- Consider adding to strategy later if you want portfolio hedges (SPX puts)

**Sources:** [QuantConnect Index Options](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/algoseek/us-index-options), [Index Options Launch](https://www.businesswire.com/news/home/20210921005212/en/%C2%A0QuantConnect-Launches-Index-Options-New-Options-Margin-Model)

---

### ❌ US Future Options ($1,200 history + $960/year updates)

**Why Skip:**
- Options on futures contracts (commodity options, index futures options)
- Different asset class from equity options
- Not relevant for your strategy

---

### ❌ International Future Options ($1,200 history + $960/year)

**Why Skip:**
- Non-US futures options
- Even further from your equity options focus

---

### ❌ OANDA Forex Data ($800 history + $200/year per resolution)

**What It Is:**
- 71 currency pairs (EURUSD, GBPUSD, etc.)
- Tick to daily resolution since 2007
- Integrated with OANDA brokerage for live trading

**Why Skip:**
- **Forex trading, not options trading**
- Different market (24/5, different volatility, geopolitical factors)
- No connection to your options Greeks strategies

**Sources:** [QuantConnect OANDA Forex](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/oanda/forex-data), [OANDA Integration](https://www.oanda.com/group/media-center/press-releases/oanda-and-quantconnect-announce-live-trading-integration/)

---

### ❌ OANDA CFD Data ($800 history + $200/year per resolution)

**Why Skip:**
- Contracts for Difference (equity CFDs, commodity CFDs)
- Forex market instrument, not options
- Not relevant for US equity options trading

---

## Recommended Configuration

### Budget: Core Essentials ($1,330/year)

```
Quiver Bundle (6 datasets):           $330/year
US Equity Option Universe Updates:    $1,000/year
────────────────────────────────────────────
TOTAL:                                 $1,330/year ($111/month)
```

**What You Get:**
- ✅ Daily Greeks/IV updates (production critical)
- ✅ Congress/Insider trading signals
- ✅ Retail/media sentiment (WSB, CNBC)
- ✅ Government contracts (bullish catalyst)

### Recommended: Full Suite ($1,935/year)

```
Quiver Bundle (6 datasets):           $330/year
Option Universe Updates:               $1,000/year
Brain Sentiment Bundle:                $550/year
Coarse Universe Updates:               $200/year
────────────────────────────────────────────
TOTAL:                                 $1,935/year ($161/month)
```

**Additional Value:**
- ✅ Professional sentiment analysis
- ✅ LLM ensemble validation
- ✅ Current universe data
- ✅ Reduced API costs

### Premium: Research Enhancement ($2,135/year)

Add if doing extensive backtesting research:

```
Core + Full Suite:                     $1,935/year
Coarse Universe History:               $500 (one-time)
────────────────────────────────────────────
TOTAL:                                 $2,435 (Year 1)
                                       $1,935/year (ongoing)
```

---

## Total Infrastructure Cost

### Budget Configuration

```
QuantConnect:
  Compute (B8-16 + R8-16 + L2-4):      $112/month ($1,344/year)
  Object Store (5GB):                  Included in compute
  Data Subscriptions:                  $1,330/year ($111/month)
────────────────────────────────────────────────────────────
TOTAL:                                 $223/month ($2,674/year)
```

### Recommended Configuration

```
QuantConnect:
  Compute (B8-16 + R8-16 + L2-4):      $112/month ($1,344/year)
  Object Store (5GB):                  Included in compute
  Data Subscriptions:                  $1,935/year ($161/month)
────────────────────────────────────────────────────────────
TOTAL:                                 $273/month ($3,279/year)
```

---

## ROI Analysis

### Break-Even Calculation

**Recommended Suite Cost:** $1,935/year

**Value Provided:**

1. **Option Universe Updates ($1,000/yr):**
   - Avoid 1 bad trade/month from stale Greeks
   - Savings: $100/month × 12 = $1,200/year
   - **ROI: 120%**

2. **Quiver Bundle ($330/yr):**
   - Catch 2 insider/congress signals/year
   - Value per signal: $200 (conservative)
   - Savings: $400/year
   - **ROI: 121%**

3. **Brain Sentiment ($550/yr):**
   - Reduce LLM API costs: $170/year
   - Better signals = 1 extra winning trade/quarter
   - Value: $170 + (4 × $100) = $570/year
   - **ROI: 104%**

4. **Universe Updates ($200/yr):**
   - Catch 1 momentum play missed with stale data
   - Value: $300/year
   - **ROI: 150%**

**Total Value:** $2,470/year
**Total Cost:** $1,935/year
**Net Benefit:** $535/year
**Overall ROI:** 128%

---

## Implementation Guide

### Phase 1: Essential (Month 1)

Start with Option Universe Updates:

```python
# In algorithm
class OptionsTradingBot(QCAlgorithm):
    def Initialize(self):
        # Subscribe to Option Universe
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Raw
        self.AddUniverse(self.OptionUniverse)

    def OptionUniverse(self, universe):
        # Filter using fresh Greeks
        return universe.IncludeWeeklys() \
                       .Strikes(-5, 5) \
                       .Expiration(7, 45) \
                       .Where(lambda x: 0.20 < abs(x.Greeks.Delta) < 0.40)
```

### Phase 2: Add Quiver (Month 2)

```python
# Add Congress Trading
self.congress = self.AddData(QuiverCongress, "AAPL").Symbol

def OnData(self, data):
    if data.ContainsKey(self.congress):
        congress = data[self.congress]

        if congress.BuyTransactions > 3:
            # Bullish signal from congress
            self.execute_bullish_spread()
```

### Phase 3: Add Brain Sentiment (Month 3)

```python
# Add Brain Sentiment
self.sentiment = self.AddData(BrainSentiment, "AAPL").Symbol

def analyze_with_ensemble(self, symbol):
    # Get Brain sentiment
    brain = self.BrainSentiment[symbol].SevenDaySentiment

    # Get LLM sentiment
    llm = self.llm_ensemble.analyze_latest_news(symbol)

    # Ensemble
    return (llm * 0.6) + (brain * 0.4)
```

---

## Dataset Details

### Quiver Quantitative

**Provider:** Founded 2020 by college students
**Quality:** High - scraped from SEC filings
**Update Frequency:** Daily
**Coverage:** 1,800 stocks (Congress), broader for others
**Reliability:** $100M+ in trades tracked
**Best For:** Directional signals, catalyst detection

**Data Fields:**
- Congress: Senator/Rep name, ticker, transaction type, amount, date
- Insiders: Exec name, position, shares, price, filing date
- WSB: Mention count, sentiment, upvotes
- Contracts: Award amount, agency, date

### Brain Company

**Provider:** Professional NLP firm
**Quality:** Very High - 2,000+ sources, 33 languages
**Update Frequency:** Daily
**Coverage:** 4,500 US stocks, 10,000 global
**Reliability:** Used by institutional traders
**Best For:** Sentiment validation, LLM enhancement

**Data Fields:**
- Sentiment score: -1 (bearish) to +1 (bullish)
- 7-day and 30-day windows
- Article count
- Source diversity score

### QuantConnect Universe Updates

**Provider:** QuantConnect + AlgoSeek
**Quality:** Exchange-grade
**Update Frequency:** Daily (market close)
**Coverage:** All US equity options
**Reliability:** Production-grade
**Best For:** Live trading, current Greeks/IV

**Data Fields:**
- Greeks: Delta, Gamma, Theta, Vega, Rho
- Implied volatility
- Open interest
- Volume
- Last price

---

## Cost Comparison

### Alternative Data Providers

| Provider | Similar Product | Cost | vs QC |
|----------|----------------|------|-------|
| **Quiver Direct** | Congress Trading | $25/month | QC: $55/year (78% cheaper) |
| **Bloomberg Terminal** | Sentiment + Insider | $24,000/year | QC: $880/year (96% cheaper) |
| **Refinitiv** | News Sentiment | $12,000/year | QC: $550/year (95% cheaper) |

**QuantConnect Advantage:** Deeply discounted for algo traders

---

## Sources

- [QuantConnect US Equity Options](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/algoseek/us-equity-options)
- [Charles Schwab API Market Data](https://developer.schwab.com/)
- [Quiver Quantitative Congress Trading](https://www.quantconnect.com/data/quiver-quantitative-congress-trading)
- [Quiver Quantitative Insider Trading](https://www.quantconnect.com/data/quiver-quantitative-insider-trading)
- [Brain Social Sentiment](https://www.quantconnect.com/data/brain-sentiment-indicator)
- [Quiver Quantitative Review](https://www.wallstreetzen.com/blog/quiver-quantitative-review/)
- [Brain Alternative Data](https://www.quantrocket.com/blog/brain-alternative-data)

---

**Recommendation:** Start with **Budget tier** ($1,330/year) for essential data, upgrade to **Full Suite** ($1,935/year) after validating ROI in first quarter.
