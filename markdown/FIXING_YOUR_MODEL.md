# Fixing Your Model: Handling NWS & Visual Crossing Forecasts

## The Problem

Your current model likely has these issues:

1. **Fixed uncertainty (±4°F)** - Doesn't adapt to forecast quality
2. **No validation of forecast agreement** - Bets even when forecasts disagree
3. **Suboptimal blending** - May not weight forecasts correctly
4. **No tracking** - Can't measure or improve performance

## Typical Forecast Accuracy (What to Expect)

### NWS (National Weather Service) ⭐ Most Reliable

```
Tomorrow (Day 1): ±2.5°F MAE
Day 2:           ±3.2°F MAE
Day 3:           ±3.8°F MAE
```

**For KLGA specifically:** ±2.0-2.5°F for tomorrow

### Visual Crossing

```
Tomorrow: ±2.6°F MAE
```

Similar to NWS, uses blend of multiple models

### Your ML Model

```
Without forecast: ±6°F MAE
With forecast:    ±3-4°F MAE
With today's temps: ±2.5-3.5°F MAE
```

## What You Should Do

### Step 1: Check Forecast Agreement

**Before betting, verify forecasts agree:**

```python
nws = 72.0
vc = 71.5
ml = 73.0

spread = max(nws, vc, ml) - min(nws, vc, ml)
# spread = 1.5°F

if spread < 3:
    print("✓ Strong agreement - safe to bet")
    uncertainty = 2.5
elif spread < 5:
    print("⚠ Moderate agreement - bet cautiously")
    uncertainty = 3.5
else:
    print("✗ Poor agreement - PASS")
    uncertainty = 5.0
```

### Step 2: Use Dynamic Uncertainty

**Don't use fixed ±4°F - calculate based on conditions:**

```python
from scripts.pipelines.improved_uncertainty import ForecastEnsemble

ensemble = ForecastEnsemble()
ensemble.add_forecast('nws', 72.0)
ensemble.add_forecast('visual_crossing', 71.5)
ensemble.add_forecast('ml_with_forecast', 73.0)

# Get consensus with dynamic uncertainty
consensus, uncertainty = ensemble.get_consensus(method='inverse_variance')
# Result: 72.0°F ± 2.1°F (lower because forecasts agree!)
```

### Step 3: Blend Forecasts Properly

**Recommended weighting:**

```python
# Method 1: Fixed weights (simple)
final = 0.50 * nws + 0.30 * vc + 0.20 * ml

# Method 2: Inverse variance (optimal)
# Weight by inverse of uncertainty squared
nws_weight = 1 / (2.5**2)  # = 0.16
vc_weight = 1 / (2.6**2)   # = 0.15
ml_weight = 1 / (3.0**2)   # = 0.11

total = nws_weight + vc_weight + ml_weight
final = (nws*nws_weight + vc*vc_weight + ml*ml_weight) / total
```

**The ensemble class does this automatically!**

### Step 4: Adjust Edge Requirements

**Require larger edge when uncertainty is high:**

```python
if uncertainty > 5:
    min_edge = 0.10  # Need 10% edge
elif uncertainty > 4:
    min_edge = 0.08  # Need 8% edge
elif uncertainty > 3:
    min_edge = 0.06  # Need 6% edge
else:
    min_edge = 0.05  # Standard 5% edge
```

### Step 5: Track Performance

**Save predictions daily:**

```python
# predictions.csv
date,nws,vc,ml,final,uncertainty,actual,error
2025-01-19,72.0,71.5,73.0,71.9,2.1,70.5,1.4
2025-01-20,68.0,67.5,69.0,67.9,2.2,68.2,-0.3
...
```

**After 30+ days, analyze:**

```python
df = pd.read_csv('predictions.csv')

print(f"Your MAE: {df['error'].abs().mean():.2f}°F")
print(f"Your RMSE: {np.sqrt((df['error']**2).mean()):.2f}°F")
print(f"Your Bias: {df['error'].mean():+.2f}°F")

# Check calibration
within_1sigma = (df['error'].abs() <= df['uncertainty']).mean()
print(f"Coverage: {within_1sigma*100:.1f}% (target: 68%)")

# If coverage is too low, increase uncertainty
# If coverage is too high, decrease uncertainty
```

## Practical Implementation

### Update Your Pipeline

Replace this:

```python
# OLD - Fixed uncertainty
if has_forecast:
    uncertainty = 4.0
else:
    uncertainty = 6.0

final_pred = 0.6 * nws_forecast + 0.4 * ml_prediction
```

With this:

```python
# NEW - Dynamic uncertainty
from scripts.pipelines.improved_uncertainty import (
    ForecastEnsemble,
    get_minimum_edge_required
)

ensemble = ForecastEnsemble()

if nws_forecast:
    ensemble.add_forecast('nws', nws_forecast)
if vc_forecast:
    ensemble.add_forecast('visual_crossing', vc_forecast)
if ml_prediction:
    ensemble.add_forecast('ml_with_forecast', ml_prediction)

# Check if forecasts agree
should_bet, reason = ensemble.should_bet(min_agreement='moderate')
if not should_bet:
    print(f"⚠ {reason} - Skipping betting")
    return

# Get consensus
final_pred, uncertainty = ensemble.get_consensus(method='inverse_variance')

# Adjust for conditions
if have_todays_temps:
    uncertainty *= 0.85

# Get minimum edge required
min_edge = get_minimum_edge_required(uncertainty)

print(f"Prediction: {final_pred:.1f}°F ± {uncertainty:.1f}°F")
print(f"Min edge required: {min_edge:.1%}")
```

## When to Use Each Forecast

### NWS Only (Most Common)

```python
ensemble = ForecastEnsemble()
ensemble.add_forecast('nws', 72.0)

consensus, unc = ensemble.get_consensus()
# Result: 72.0°F ± 2.5°F
```

### NWS + Visual Crossing (Best)

```python
ensemble = ForecastEnsemble()
ensemble.add_forecast('nws', 72.0)
ensemble.add_forecast('visual_crossing', 71.5)

consensus, unc = ensemble.get_consensus()
# Result: 71.8°F ± 1.8°F (lower uncertainty!)
```

### NWS + VC + ML (Maximum Info)

```python
ensemble = ForecastEnsemble()
ensemble.add_forecast('nws', 72.0)
ensemble.add_forecast('visual_crossing', 71.5)
ensemble.add_forecast('ml_with_forecast', 73.0)

consensus, unc = ensemble.get_consensus()
# Result: 72.0°F ± 2.1°F
```

### Only ML Model (Fallback)

```python
ensemble = ForecastEnsemble()
ensemble.add_forecast('ml_model', 73.0)

consensus, unc = ensemble.get_consensus()
# Result: 73.0°F ± 4.0°F (higher uncertainty)
```

## Red Flags: When NOT to Bet

### 1. Forecasts Disagree

```python
nws = 72.0
vc = 68.0  # 4°F difference!

# PASS - Wait for convergence
```

### 2. High Uncertainty

```python
if uncertainty > 5.0:
    # Too uncertain - edge needs to be huge
    # Better to pass
```

### 3. Rapid Weather Changes

```python
if abs(today_high - yesterday_high) > 15:
    # Unstable pattern
    uncertainty *= 1.5
```

### 4. Extreme Temperatures

```python
if forecast < 10 or forecast > 95:
    # Model may not have enough training data
    uncertainty *= 1.3
```

## Expected Results

### With Proper Uncertainty:

**Before (Fixed ±4°F):**

- Betting on marginal opportunities
- Overconfident in predictions
- Lower win rate

**After (Dynamic uncertainty):**

- Only bet when forecasts agree
- Appropriate confidence levels
- Higher win rate
- Better risk management

### Example Comparison:

**Scenario: Forecasts disagree**

```
NWS: 72°F
VC:  68°F
ML:  70°F
```

**Old approach:**

- Final: 70.4°F ± 4°F
- Bets on markets near 70°F
- High risk due to disagreement

**New approach:**

- Final: 70.0°F ± 4.8°F (increased due to disagreement)
- Requires 8% edge instead of 5%
- May PASS if agreement is poor
- Better risk management

## Quick Start

1. **Install the improved uncertainty module:**

   ```bash
   # Already created: scripts/pipelines/improved_uncertainty.py
   ```

2. **Test it:**

   ```bash
   python scripts/pipelines/improved_uncertainty.py
   ```

3. **Update your pipeline:**
   - Replace fixed uncertainty with `ForecastEnsemble`
   - Add forecast agreement checks
   - Use dynamic edge requirements

4. **Start tracking:**
   - Save predictions daily
   - Record actual temperatures
   - Analyze after 30+ days

5. **Iterate:**
   - Adjust weights based on performance
   - Calibrate uncertainty
   - Improve over time

## Bottom Line

**Your model isn't necessarily wrong - you just need to:**

1. ✅ Use dynamic uncertainty based on forecast agreement
2. ✅ Check that forecasts agree before betting
3. ✅ Blend forecasts optimally (inverse variance weighting)
4. ✅ Require larger edge when uncertainty is high
5. ✅ Track performance and iterate

**Expected uncertainty for tomorrow:**

- **Best case** (all forecasts agree): ±2.0-2.5°F
- **Normal case** (moderate agreement): ±2.5-3.5°F
- **Worst case** (forecasts disagree): ±4.0-5.0°F or PASS

Use the `ForecastEnsemble` class to handle all of this automatically!
