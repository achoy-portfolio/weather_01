# How the Pipeline Calculates Betting Decisions

## Overview

The pipeline uses **Expected Value (EV)** analysis combined with **Kelly Criterion** position sizing to identify profitable betting opportunities on Polymarket temperature markets.

## Core Concepts

### 1. Expected Value (EV)

Expected Value measures the average profit/loss per dollar bet over the long run.

**Formula:**

```
EV = (Model Probability × Payout Multiplier) - 1
```

**Example:**

- Market odds: 40% (implies you get paid $2.50 for every $1 if you win)
- Your model probability: 50%
- Payout multiplier: 1 / 0.40 = 2.5
- EV = (0.50 × 2.5) - 1 = 0.25 = **+25% EV**

This means for every $1 you bet, you expect to make $0.25 profit on average.

### 2. Edge

Edge is the difference between your model's probability and the market's probability.

**Formula:**

```
Edge = Model Probability - Market Probability
```

**Example:**

- Model: 50%
- Market: 40%
- Edge = 50% - 40% = **+10% edge**

### 3. Kelly Criterion

Kelly Criterion calculates the optimal bet size to maximize long-term growth while managing risk.

**Formula:**

```
Kelly Fraction = (b × p - q) / b

Where:
- b = decimal_odds - 1 (net odds)
- p = probability of winning (your model)
- q = 1 - p (probability of losing)
```

**Safety: Fractional Kelly**
The pipeline uses **Quarter Kelly (25%)** to reduce risk:

```
Bet Size = Kelly Fraction × 0.25 × Bankroll
```

## Pipeline Decision Process

### Step 1: Generate Model Prediction

The pipeline predicts tomorrow's peak temperature using:

1. **Historical data** (last 30 days)
2. **Today's intraday temperatures** (6am, 9am, noon, 3pm)
3. **NWS forecast** (if available)

**Blended Prediction:**

```python
final_pred = (0.6 × NWS_forecast) + (0.4 × ML_model)
```

**Uncertainty Estimation:**

- Uses probabilistic model (quantile regression) to estimate ±uncertainty
- Typical uncertainty: ±4-6°F

### Step 2: Calculate Probabilities for Each Threshold

For each Polymarket market (e.g., "Will temp be > 75°F?"):

**Model Probability:**

```python
z_score = (threshold - prediction) / uncertainty
model_prob = 1 - norm.cdf(z_score)  # Using normal distribution
```

**Example:**

- Prediction: 72°F ± 5°F
- Threshold: 75°F
- z_score = (75 - 72) / 5 = 0.6
- model_prob = 1 - 0.7257 = **27.4%**

### Step 3: Compare to Market Odds

**Market Probability:**

- Extracted from Polymarket's current odds
- Example: Market shows 40% probability

**Calculate Edge:**

```
Edge = 27.4% - 40% = -12.6%
```

In this case, the market is **overpricing** this outcome, so we **PASS**.

### Step 4: Calculate Expected Value

```python
payout_multiplier = 1 / market_odds
expected_value = (model_prob × payout_multiplier) - 1
ev_pct = expected_value × 100
```

**Example (Positive EV):**

- Market odds: 30%
- Model prob: 45%
- Payout: 1/0.30 = 3.33x
- EV = (0.45 × 3.33) - 1 = 0.50 = **+50% EV** ✅

**Example (Negative EV):**

- Market odds: 60%
- Model prob: 45%
- Payout: 1/0.60 = 1.67x
- EV = (0.45 × 1.67) - 1 = -0.25 = **-25% EV** ❌

### Step 5: Calculate Kelly Bet Size

```python
decimal_odds = 1 / market_odds
b = decimal_odds - 1
p = model_prob
q = 1 - model_prob

kelly = (b × p - q) / b
kelly_fractional = kelly × 0.25  # Quarter Kelly for safety
bet_size = kelly_fractional × bankroll
```

**Example:**

- Bankroll: $1,000
- Market odds: 30% (decimal odds = 3.33)
- Model prob: 45%
- b = 3.33 - 1 = 2.33
- kelly = (2.33 × 0.45 - 0.55) / 2.33 = 0.214
- kelly_fractional = 0.214 × 0.25 = 0.0535
- **Bet size = $53.50**

### Step 6: Make Recommendation

**Betting Criteria:**

```python
if expected_value > 0.05:  # EV > 5%
    recommendation = "BET"
else:
    recommendation = "PASS"
```

**Why 5% threshold?**

- Accounts for model uncertainty
- Provides margin of safety
- Ensures only strong opportunities are bet

## Complete Example

### Scenario

- **Prediction:** 72°F ± 5°F
- **Bankroll:** $1,000
- **Market:** "Will temp be > 70°F?"
- **Market odds:** 55%

### Calculations

**1. Model Probability:**

```
z_score = (70 - 72) / 5 = -0.4
model_prob = 1 - norm.cdf(-0.4) = 65.5%
```

**2. Edge:**

```
Edge = 65.5% - 55% = +10.5%
```

**3. Expected Value:**

```
Payout = 1 / 0.55 = 1.82x
EV = (0.655 × 1.82) - 1 = 0.192 = +19.2% EV ✅
```

**4. Kelly Bet Size:**

```
b = 1.82 - 1 = 0.82
kelly = (0.82 × 0.655 - 0.345) / 0.82 = 0.234
quarter_kelly = 0.234 × 0.25 = 0.0585
Bet size = $58.50
```

**5. Recommendation:**

```
✅ BET YES on "> 70°F" at 55%
Amount: $58.50
Edge: +10.5%
EV: +19.2%
```

## Risk Management

### 1. Fractional Kelly (25%)

- Reduces volatility
- Protects against model errors
- More conservative than full Kelly

### 2. Minimum Edge Requirement (5%)

- Only bet when edge is significant
- Accounts for model uncertainty
- Reduces false positives

### 3. Diversification

- Multiple small bets across different thresholds
- Reduces impact of any single bet
- Smooths returns

### 4. Bankroll Management

- Never bet more than Kelly suggests
- Total allocation typically 5-15% of bankroll
- Preserves capital for future opportunities

## Dashboard Display

The dashboard shows:

1. **Prediction Summary**
   - Point estimate (e.g., 72°F)
   - Uncertainty range (e.g., ±5°F)
   - 80% confidence interval

2. **Market Opportunities**
   - Each threshold analyzed
   - Market odds vs Model probability
   - Edge and EV calculations
   - Kelly bet size

3. **Recommendations**
   - Clear BET/PASS decisions
   - Which side to bet (YES/NO)
   - Exact bet amount
   - Expected profit

4. **Technical Analysis**
   - Detailed probability calculations
   - Risk/reward metrics
   - Historical performance tracking

## Key Insights

### When to Bet

✅ **Positive EV** (EV > 5%)
✅ **Significant edge** (>5% difference)
✅ **High confidence** (low uncertainty)
✅ **Liquid market** (high volume)

### When to Pass

❌ **Negative EV** (EV < 0%)
❌ **Small edge** (<5% difference)
❌ **High uncertainty** (±10°F or more)
❌ **Illiquid market** (low volume)

### Common Scenarios

**Scenario 1: Market Underpricing**

- Market: 30% | Model: 45%
- Result: **BET YES** (market too pessimistic)

**Scenario 2: Market Overpricing**

- Market: 70% | Model: 55%
- Result: **BET NO** (market too optimistic)

**Scenario 3: Fair Market**

- Market: 50% | Model: 52%
- Result: **PASS** (edge too small)

## Mathematical Foundation

The strategy is based on:

1. **Law of Large Numbers**: Over many bets, results converge to expected value
2. **Kelly Criterion**: Optimal bet sizing for long-term growth
3. **Normal Distribution**: Temperature follows approximately normal distribution
4. **Bayesian Updating**: Combines multiple information sources (model + forecast)

## Performance Tracking

The pipeline saves results to track:

- Actual vs predicted temperatures
- Bet outcomes (win/loss)
- Realized EV vs expected EV
- Model calibration
- ROI over time

This allows continuous improvement of the model and strategy.
