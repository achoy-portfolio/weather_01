# Forecast Error Model - Documentation

## Overview

This directory contains the statistical analysis of forecast errors for the 9 PM forecast predicting next day's maximum temperature.

## Files

### 1. `error_distribution_analysis.json`

Complete statistical analysis of forecast errors including:

- **Basic Statistics**: mean, median, std, min, max
- **Percentiles**: 5th, 10th, 25th, 50th, 75th, 90th, 95th
- **Normality Tests**: Shapiro-Wilk test results
- **Distribution Fit**: How well errors match normal distribution
- **Confidence Intervals**: Ranges for 50%, 68%, 90%, 95% confidence

### 2. `forecast_accuracy_metrics.json`

Hourly and daily forecast accuracy metrics:

- **Metric 1**: Hourly temperature accuracy (MAE: 1.96°F)
- **Metric 2**: Next day maximum accuracy (MAE: 2.17°F)

### 3. `error_model.json` (legacy)

Original error model with lead time and seasonal breakdowns.

## Key Findings

### Forecast Accuracy (9 PM for Next Day)

- **MAE**: 2.17°F (average error)
- **RMSE**: 2.66°F
- **Bias**: +0.61°F (forecasts are slightly warm)
- **Sample Size**: 361 days

### Error Distribution

- **Is Normal**: Yes, approximately
- **Mean Error**: +0.61°F (warm bias)
- **Std Deviation**: 2.58°F

### Confidence Intervals

For a forecast of X°F, the actual temperature will be:

- **50% of the time**: within ±1.9°F
- **68% of the time**: within ±2.7°F
- **90% of the time**: within ±4.3°F
- **95% of the time**: within ±5.2°F

## How to Use

### Loading the Error Model

```python
import json

with open('data/processed/error_distribution_analysis.json', 'r') as f:
    error_model = json.load(f)

mean_error = error_model['basic_statistics']['mean']  # 0.61°F
std_error = error_model['basic_statistics']['std']    # 2.58°F
```

### Calculating Probabilities

```python
from scipy import stats

def prob_above_threshold(forecast, threshold, mean_error, std_error):
    """Calculate P(actual >= threshold | forecast)"""
    adjusted_forecast = forecast - mean_error  # Account for bias
    z_score = (threshold - adjusted_forecast) / std_error
    return 1 - stats.norm.cdf(z_score)

# Example: Forecast is 45°F, what's P(actual >= 44°F)?
prob = prob_above_threshold(45, 44, 0.61, 2.58)
print(f"Probability: {prob:.1%}")  # ~55.9%
```

### Getting Confidence Intervals

```python
def confidence_interval(forecast, confidence_level, mean_error, std_error):
    """Get confidence interval for actual temperature"""
    adjusted = forecast - mean_error
    z = stats.norm.ppf((1 + confidence_level) / 2)
    margin = z * std_error
    return (adjusted - margin, adjusted + margin)

# Example: 68% confidence interval for 45°F forecast
lower, upper = confidence_interval(45, 0.68, 0.61, 2.58)
print(f"68% CI: {lower:.1f}°F to {upper:.1f}°F")  # 41.8°F to 47.0°F
```

## Betting Strategy Recommendations

### ✅ DO:

1. **Use 9 PM forecasts** for next day's high (most accurate)
2. **Account for warm bias** (subtract 0.61°F from forecast)
3. **Use normal distribution** for probability calculations
4. **Require 5%+ expected value** before betting

### ❌ DON'T:

1. **Don't use same-day forecasts** for daily max (8.26°F error)
2. **Don't ignore bias** (forecasts are systematically warm)
3. **Don't bet without edge** (need model prob > market prob)

## Example Betting Decision

**Scenario:**

- Forecast: 45°F (9 PM for tomorrow)
- Bet: Temperature ≥44°F
- Market Odds: 55% (implied probability)

**Analysis:**

1. Adjust for bias: 45 - 0.61 = 44.4°F
2. Calculate model probability: 55.9%
3. Calculate edge: 55.9% - 55% = 0.9%
4. Calculate EV: (0.559 × 1.82) - 1 = 1.7%
5. **Decision**: PASS (EV < 5% threshold)

## Updating the Model

To regenerate these files with new data:

```bash
# Run forecast accuracy analysis
python scripts/analysis/forecast_accuracy_analysis.py

# Run error distribution analysis
python scripts/analysis/check_error_distribution.py
```

## Validation

The error model has been validated to be approximately normally distributed:

- ✅ Within ±1 std: 64.0% (expected 68.3%)
- ✅ Within ±2 std: 95.6% (expected 95.4%)
- ✅ Within ±3 std: 99.4% (expected 99.7%)

## Last Updated

2026-01-18

## Sample Size

361 days of 9 PM forecasts for next day's maximum temperature
