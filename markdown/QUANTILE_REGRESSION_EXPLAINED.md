# Quantile Regression & Probability Estimation Explained

## What is Quantile Regression?

**Quantile regression** predicts different percentiles of the outcome distribution, not just the mean. This gives you a complete picture of uncertainty.

### Traditional Regression vs Quantile Regression

**Traditional Regression (Mean):**

```
Prediction: 72°F
```

You only know the average, not the uncertainty.

**Quantile Regression:**

```
Q10 (10th percentile): 67°F
Q25 (25th percentile): 69°F
Q50 (50th percentile/median): 72°F
Q75 (75th percentile): 75°F
Q90 (90th percentile): 77°F
```

You know the full distribution of possible outcomes!

## Does It Assume Normal Distribution?

**Short Answer: No, but it uses normal approximation for convenience.**

### How It Actually Works

#### Step 1: Train Separate Models for Each Quantile

The system trains **5 separate XGBoost models**, each predicting a different quantile:

```python
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

for q in quantiles:
    model = xgb.XGBRegressor(
        objective='reg:quantileerror',  # Quantile loss function
        quantile_alpha=q,                # Which quantile to predict
        ...
    )
    model.fit(X_train, y_train)
```

**Key Point:** Each model learns to predict a specific percentile **without assuming any distribution**. The model learns the actual shape from the data.

#### Step 2: Get Predictions for All Quantiles

For tomorrow's weather, predict all 5 quantiles:

```python
Q10 = model_10.predict(features)  # 67°F
Q25 = model_25.predict(features)  # 69°F
Q50 = model_50.predict(features)  # 72°F (median)
Q75 = model_75.predict(features)  # 75°F
Q90 = model_90.predict(features)  # 77°F
```

This gives you the **empirical distribution** learned from historical data.

#### Step 3: Normal Approximation (For Convenience)

To calculate probabilities like P(temp > 75°F), the code uses a **normal approximation**:

```python
# Estimate standard deviation from quantiles
# For normal distribution: Q90 - Q10 ≈ 2.56 × std
estimated_std = (Q90 - Q10) / 2.56
estimated_std = (77 - 67) / 2.56 = 3.9°F

# Calculate probability using normal CDF
z_score = (threshold - Q50) / estimated_std
z_score = (75 - 72) / 3.9 = 0.77

prob_above = 1 - norm.cdf(0.77) = 22%
```

### Why Use Normal Approximation?

**Pros:**

1. **Simple** - Easy to calculate P(temp > threshold) for any threshold
2. **Smooth** - Gives continuous probabilities
3. **Reasonable** - Temperature distributions are often approximately normal
4. **Fast** - No need for complex interpolation

**Cons:**

1. **Not exact** - Real distribution may have skewness or fat tails
2. **Loses information** - Only uses 3 quantiles (Q10, Q50, Q90)

## Alternative: Direct Quantile Interpolation (More Accurate)

Instead of assuming normal distribution, you could **interpolate directly** between quantiles:

### Example: P(temp > 75°F)

Given quantiles:

```
Q10 = 67°F (10% below, 90% above)
Q25 = 69°F (25% below, 75% above)
Q50 = 72°F (50% below, 50% above)
Q75 = 75°F (75% below, 25% above)  ← 75°F is exactly Q75!
Q90 = 77°F (90% below, 10% above)
```

**Direct answer:** P(temp > 75°F) = **25%** (since 75°F is the 75th percentile)

### For Thresholds Between Quantiles

If threshold = 73°F (between Q50=72°F and Q75=75°F):

**Linear interpolation:**

```
Q50 = 72°F → 50% above
Q75 = 75°F → 25% above

73°F is 1/3 of the way from 72 to 75
Interpolated probability = 50% - (1/3 × (50% - 25%))
                        = 50% - 8.3%
                        = 41.7%
```

This is **more accurate** than normal approximation but requires more code.

## Comparison: Normal vs Direct Interpolation

### Example Scenario

```
Q10 = 67°F
Q50 = 72°F
Q90 = 77°F
Threshold = 75°F
```

**Method 1: Normal Approximation (Current)**

```
std = (77 - 67) / 2.56 = 3.9°F
z = (75 - 72) / 3.9 = 0.77
P(temp > 75) = 1 - norm.cdf(0.77) = 22%
```

**Method 2: Direct Interpolation**

```
75°F is between Q50 (72°F) and Q90 (77°F)
75 is 60% of the way from 72 to 77
P(temp > 75) = 50% - 0.6 × (50% - 10%) = 26%
```

**Difference:** 4 percentage points

## When Does Normal Approximation Work Well?

### Good Cases ✅

- **Symmetric distributions** (temperature is often symmetric)
- **Moderate temperatures** (middle of the range)
- **Stable weather patterns** (low uncertainty)

### Poor Cases ❌

- **Extreme temperatures** (tails of distribution)
- **Skewed distributions** (e.g., heat waves)
- **Bimodal patterns** (e.g., cold front passing through)

## Validation: Coverage Testing

The model checks if predictions are well-calibrated:

```python
# 50% interval should contain 50% of actual values
coverage_50 = mean((actual >= Q25) & (actual <= Q75))
# Target: 50%, Actual: ~48-52% (good!)

# 80% interval should contain 80% of actual values
coverage_80 = mean((actual >= Q10) & (actual <= Q90))
# Target: 80%, Actual: ~78-82% (good!)
```

If coverage matches targets, the quantile predictions are **well-calibrated** regardless of distribution shape.

## Real Example from Code

```python
# Actual predictions from model
Q10 = 67.2°F
Q25 = 69.5°F
Q50 = 72.1°F
Q75 = 74.8°F
Q90 = 77.3°F

# Calculate probabilities for betting markets
thresholds = [65, 70, 75, 80]

for threshold in thresholds:
    std = (77.3 - 67.2) / 2.56 = 3.95°F
    z = (threshold - 72.1) / 3.95
    prob = 1 - norm.cdf(z)

    print(f"P(temp > {threshold}°F) = {prob:.1%}")
```

**Output:**

```
P(temp > 65°F) = 96.4%  (very likely)
P(temp > 70°F) = 70.2%  (likely)
P(temp > 75°F) = 22.8%  (unlikely)
P(temp > 80°F) = 2.3%   (very unlikely)
```

## Advantages of Quantile Regression

### 1. No Distribution Assumptions

- Learns actual distribution from data
- Handles skewness, fat tails, multimodality
- More flexible than parametric models

### 2. Robust to Outliers

- Median (Q50) is more robust than mean
- Extreme values don't distort predictions

### 3. Full Uncertainty Quantification

- Not just point estimate + error bars
- Complete picture of possible outcomes
- Better risk assessment

### 4. Calibrated Predictions

- Coverage testing ensures reliability
- Probabilities match actual frequencies
- Trustworthy for betting decisions

## Improving the Current Implementation

### Option 1: Use More Quantiles

```python
quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
```

More quantiles = better resolution of distribution

### Option 2: Direct Interpolation

```python
def prob_above_threshold(quantile_preds, threshold):
    """Calculate P(temp > threshold) by interpolating quantiles."""

    # Find quantiles bracketing the threshold
    for i, (q, temp) in enumerate(quantile_preds.items()):
        if temp >= threshold:
            if i == 0:
                return 1 - q  # Above highest quantile

            # Interpolate between quantiles
            q_low, temp_low = list(quantile_preds.items())[i-1]
            q_high, temp_high = q, temp

            # Linear interpolation
            frac = (threshold - temp_low) / (temp_high - temp_low)
            prob_below = q_low + frac * (q_high - q_low)
            return 1 - prob_below

    return 0  # Below lowest quantile
```

### Option 3: Fit Flexible Distribution

```python
from scipy.stats import skewnorm

# Fit skewed normal to quantiles
params = fit_skewnorm_to_quantiles(Q10, Q50, Q90)
prob = 1 - skewnorm.cdf(threshold, *params)
```

## Key Takeaways

1. **Quantile regression does NOT assume normal distribution** - it learns the actual distribution from data

2. **Normal approximation is used for convenience** - to calculate probabilities between quantiles

3. **The approximation works well** when:
   - Temperature distributions are roughly symmetric
   - You're not in extreme tails
   - Coverage testing shows good calibration

4. **You could improve accuracy** by:
   - Using direct quantile interpolation
   - Training more quantile models
   - Fitting flexible distributions (skewed normal, etc.)

5. **Current approach is pragmatic** - balances accuracy, simplicity, and computational efficiency

## Bottom Line

The quantile regression model **learns the true distribution** from historical data without assumptions. The normal approximation is just a **convenient tool** for calculating probabilities, not a fundamental assumption. The model's predictions are validated through coverage testing to ensure they're reliable for betting decisions.
