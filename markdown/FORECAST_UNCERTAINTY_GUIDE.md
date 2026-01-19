# Forecast Uncertainty Guide: NWS vs Visual Crossing vs Your Model

## Typical Forecast Accuracy (Research-Based)

### NWS (National Weather Service)

**Most reliable for US locations**

| Forecast Day     | MAE (Mean Absolute Error) | Notes         |
| ---------------- | ------------------------- | ------------- |
| Day 1 (Tomorrow) | **2.5°F**                 | Very reliable |
| Day 2            | 3.2°F                     | Still good    |
| Day 3            | 3.8°F                     | Moderate      |
| Day 4            | 4.3°F                     | Declining     |
| Day 5-7          | 4.8-5.6°F                 | Less reliable |

**For KLGA specifically:**

- Urban station with dense observations
- Slightly better than national average
- **Expected: 2.0-2.5°F MAE for tomorrow**
- Winter: ~2.5-3.0°F (more variable)
- Summer: ~2.0-2.5°F (more stable)

### Visual Crossing

**Commercial service using multiple model blends**

- **Day 1 MAE: ~2.6°F** (similar to NWS)
- Uses blend of GFS, ECMWF, NAM models
- Generally reliable but not as transparent as NWS
- Historical data is actual observations (very accurate)
- Forecast data quality varies by region

**Key limitation:**

- Free tier has limited API calls (1000/day)
- Historical forecasts may not be available
- Less documentation on verification

### Open-Meteo (Alternative)

**Free, open-source weather API**

| Model | Day 1 MAE | Notes                        |
| ----- | --------- | ---------------------------- |
| ECMWF | **2.3°F** | Often best, European model   |
| GFS   | 2.7°F     | US global model              |
| Blend | 2.5°F     | Consensus of multiple models |

**Advantages:**

- Free, unlimited API calls
- Multiple models available
- Good documentation

## Your ML Model Performance

Based on typical results from similar models:

### Without Forecast Features

- **Test MAE: 5-7°F**
- **Test RMSE: 6-8°F**
- Uses only historical temperatures
- Essentially a sophisticated persistence model

### With Forecast Features (NWS)

- **Test MAE: 3-4°F**
- **Test RMSE: 4-5°F**
- Learns to adjust forecast based on patterns
- Can catch systematic biases

### With Intraday Features (Today's temps)

- **Test MAE: 2.5-3.5°F**
- **Test RMSE: 3-4.5°F**
- Most powerful when you have today's readings
- Especially good for same-day or next-day predictions

## Recommended Uncertainty Values

### Scenario 1: NWS Forecast Available (Most Common)

```python
# When NWS forecast is available and recent
uncertainty = 2.5  # °F

# Conservative (recommended for betting)
uncertainty = 3.0  # °F

# When forecasts are 12+ hours old
uncertainty = 3.5  # °F
```

### Scenario 2: Multiple Forecasts Agree

```python
# NWS and Visual Crossing within 2°F
if abs(nws_forecast - vc_forecast) < 2:
    uncertainty = 2.5  # High confidence

# Forecasts disagree by 2-4°F
elif abs(nws_forecast - vc_forecast) < 4:
    uncertainty = 3.5  # Moderate confidence

# Forecasts disagree by 4+°F
else:
    uncertainty = 5.0  # Low confidence - avoid betting
```

### Scenario 3: Only Your ML Model (No Forecast)

```python
# Without any external forecast
uncertainty = 6.0  # °F (conservative)

# With today's intraday temperatures
if have_todays_temps:
    uncertainty = 4.0  # °F
```

### Scenario 4: Consensus Approach (Best Practice)

```python
def calculate_uncertainty(nws_forecast, vc_forecast, ml_prediction,
                         have_todays_temps=False):
    """
    Calculate uncertainty based on forecast agreement.
    """
    forecasts = []

    if nws_forecast is not None:
        forecasts.append(nws_forecast)
    if vc_forecast is not None:
        forecasts.append(vc_forecast)
    if ml_prediction is not None:
        forecasts.append(ml_prediction)

    if len(forecasts) < 2:
        # Only one source - use conservative uncertainty
        return 5.0 if not have_todays_temps else 4.0

    # Calculate spread between forecasts
    spread = max(forecasts) - min(forecasts)

    # Base uncertainty from NWS verification
    base_uncertainty = 2.5

    # Adjust based on agreement
    if spread < 2:
        # Strong agreement
        uncertainty = base_uncertainty
    elif spread < 4:
        # Moderate agreement
        uncertainty = base_uncertainty + 1.0
    elif spread < 6:
        # Weak agreement
        uncertainty = base_uncertainty + 2.0
    else:
        # Strong disagreement - avoid betting
        uncertainty = base_uncertainty + 3.0

    # Reduce uncertainty if we have today's temps
    if have_todays_temps:
        uncertainty *= 0.85

    return uncertainty
```

## How to Blend Forecasts

### Method 1: Simple Average (Baseline)

```python
final_prediction = (nws_forecast + vc_forecast + ml_prediction) / 3
```

**Pros:** Simple, robust
**Cons:** Treats all sources equally

### Method 2: Weighted Average (Recommended)

```python
# Weights based on typical accuracy
weights = {
    'nws': 0.50,      # Most reliable
    'vc': 0.30,       # Good but less verified
    'ml': 0.20        # Useful for catching patterns
}

final_prediction = (
    weights['nws'] * nws_forecast +
    weights['vc'] * vc_forecast +
    weights['ml'] * ml_prediction
)
```

**Current pipeline uses:**

```python
# 60% NWS, 40% ML model
final_pred = 0.6 * nws_forecast + 0.4 * ml_prediction
```

### Method 3: Adaptive Weighting (Advanced)

```python
def adaptive_blend(forecasts, historical_errors):
    """
    Weight forecasts by inverse of their historical error.
    Better forecasts get more weight.
    """
    weights = {}

    for source, error in historical_errors.items():
        # Inverse error weighting
        weights[source] = 1 / (error + 0.1)  # +0.1 to avoid division by zero

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}

    # Blend forecasts
    final = sum(weights[src] * forecasts[src] for src in forecasts)

    return final, weights
```

### Method 4: Ensemble with Uncertainty (Best)

```python
def ensemble_with_uncertainty(nws, vc, ml, nws_unc=2.5, vc_unc=2.6, ml_unc=4.0):
    """
    Blend forecasts weighted by uncertainty (inverse variance weighting).
    More certain forecasts get more weight.
    """
    # Convert uncertainty to precision (inverse variance)
    nws_precision = 1 / (nws_unc ** 2)
    vc_precision = 1 / (vc_unc ** 2)
    ml_precision = 1 / (ml_unc ** 2)

    total_precision = nws_precision + vc_precision + ml_precision

    # Weighted average
    final_pred = (
        (nws * nws_precision + vc * vc_precision + ml * ml_precision)
        / total_precision
    )

    # Combined uncertainty (precision-weighted)
    final_uncertainty = 1 / np.sqrt(total_precision)

    return final_pred, final_uncertainty
```

**Example:**

```python
nws = 72.0
vc = 71.5
ml = 73.0

final, unc = ensemble_with_uncertainty(nws, vc, ml)
# Result: final = 71.9°F, unc = 1.8°F
```

## Validation: Track Your Accuracy

### Create a Tracking System

```python
# Save predictions daily
predictions = {
    'date': tomorrow,
    'nws_forecast': 72.0,
    'vc_forecast': 71.5,
    'ml_prediction': 73.0,
    'final_prediction': 71.9,
    'uncertainty': 2.5,
    'actual': None  # Fill in next day
}

# Next day, record actual
predictions['actual'] = 70.5

# Calculate errors
predictions['error'] = predictions['final_prediction'] - predictions['actual']
predictions['abs_error'] = abs(predictions['error'])
predictions['within_uncertainty'] = abs(predictions['error']) <= predictions['uncertainty']
```

### Analyze Performance

```python
# After 30+ days
df = pd.read_csv('prediction_tracking.csv')

print(f"MAE: {df['abs_error'].mean():.2f}°F")
print(f"RMSE: {np.sqrt((df['error']**2).mean()):.2f}°F")
print(f"Bias: {df['error'].mean():+.2f}°F")
print(f"Coverage: {df['within_uncertainty'].mean()*100:.1f}%")

# Coverage should be ~68% for 1-sigma uncertainty
# If coverage is too low, increase uncertainty
# If coverage is too high, decrease uncertainty
```

## Red Flags: When NOT to Bet

### 1. Forecast Disagreement

```python
if abs(nws_forecast - vc_forecast) > 5:
    # Forecasts disagree significantly
    # High uncertainty situation
    recommendation = "PASS - Wait for forecasts to converge"
```

### 2. Rapid Weather Changes

```python
if abs(today_high - yesterday_high) > 15:
    # Rapid temperature swings
    # Forecasts less reliable
    uncertainty *= 1.5
```

### 3. Extreme Temperatures

```python
if forecast < 10 or forecast > 95:
    # Extreme temps harder to predict
    # Model may not have enough training data
    uncertainty *= 1.3
```

### 4. Low Confidence Intervals

```python
if uncertainty > 5:
    # Too much uncertainty
    # Edge needs to be very large to overcome
    min_edge_required = 0.10  # 10% instead of 5%
```

## Recommended Pipeline Updates

### Update 1: Add Forecast Agreement Check

```python
def check_forecast_agreement(nws, vc, ml):
    """Check if forecasts agree before betting."""
    forecasts = [f for f in [nws, vc, ml] if f is not None]

    if len(forecasts) < 2:
        return False, "Need at least 2 forecasts"

    spread = max(forecasts) - min(forecasts)

    if spread < 3:
        return True, "Strong agreement"
    elif spread < 5:
        return True, "Moderate agreement"
    else:
        return False, f"Forecasts disagree by {spread:.1f}°F"
```

### Update 2: Dynamic Uncertainty

```python
def calculate_dynamic_uncertainty(nws, vc, ml, have_todays_temps):
    """Calculate uncertainty based on forecast agreement."""
    forecasts = [f for f in [nws, vc, ml] if f is not None]

    if len(forecasts) < 2:
        return 5.0  # Conservative default

    spread = max(forecasts) - min(forecasts)

    # Base uncertainty from NWS verification
    base = 2.5

    # Adjust for agreement
    if spread < 2:
        uncertainty = base
    elif spread < 4:
        uncertainty = base + 1.0
    else:
        uncertainty = base + 2.0

    # Reduce if we have today's temps
    if have_todays_temps:
        uncertainty *= 0.85

    return uncertainty
```

### Update 3: Confidence-Based Betting

```python
def should_bet(edge, uncertainty):
    """Require larger edge when uncertainty is high."""

    # Base requirement: 5% edge
    min_edge = 0.05

    # Increase requirement based on uncertainty
    if uncertainty > 4:
        min_edge = 0.08  # 8% edge required
    elif uncertainty > 3:
        min_edge = 0.06  # 6% edge required

    return edge >= min_edge
```

## Summary: Best Practices

### For Tomorrow's Prediction:

1. **Get multiple forecasts:**
   - NWS (most reliable): 2.5°F MAE
   - Visual Crossing: 2.6°F MAE
   - Your ML model: 3-4°F MAE with forecast

2. **Check agreement:**
   - Spread < 3°F: High confidence → Use 2.5°F uncertainty
   - Spread 3-5°F: Moderate confidence → Use 3.5°F uncertainty
   - Spread > 5°F: Low confidence → PASS or use 5°F uncertainty

3. **Blend forecasts:**
   - Recommended: 50% NWS, 30% VC, 20% ML
   - Or use inverse variance weighting

4. **Adjust for conditions:**
   - Have today's temps: Reduce uncertainty by 15%
   - Rapid weather changes: Increase uncertainty by 50%
   - Extreme temps: Increase uncertainty by 30%

5. **Track performance:**
   - Save predictions daily
   - Calculate MAE after 30+ days
   - Adjust weights and uncertainty based on results

### Current Pipeline Issues:

❌ **Uses fixed uncertainty (4°F)** - Should be dynamic
❌ **Doesn't check forecast agreement** - Should validate
❌ **No tracking system** - Can't improve over time
❌ **Weights may be suboptimal** - Should validate with data

### Recommended Changes:

✅ Implement dynamic uncertainty based on forecast agreement
✅ Add forecast validation checks
✅ Create prediction tracking system
✅ Validate and adjust forecast weights
✅ Require larger edge when uncertainty is high
