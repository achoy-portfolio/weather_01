"""Debug probability calculation for 32-33°F range"""

import json
from scipy import stats

# Load error model
with open('data/processed/error_distribution_analysis.json', 'r') as f:
    model = json.load(f)

error_model_0d = model['by_lead_time']['0d']
error_model_1d = model['by_lead_time']['1d']

print("="*70)
print("PROBABILITY CALCULATION DEBUG")
print("="*70)

# Scenario
forecast_temp = 32.0
threshold_str = "32-33"
threshold_value = 32.5  # Midpoint
threshold_type = "range"

print(f"\nScenario:")
print(f"  Forecast Max: {forecast_temp}°F")
print(f"  Market Threshold: {threshold_str}°F")
print(f"  Threshold Type: {threshold_type}")
print(f"  Threshold Value (midpoint): {threshold_value}°F")

# For range, Polymarket rounds to whole degrees
# So 32-33 means actual temp rounds to 32 or 33
# 32 rounds from [31.5, 32.5)
# 33 rounds from [32.5, 33.5)
# Combined: [31.5, 33.5)
low = 31.5
high = 33.5

print(f"\nRange interpretation:")
print(f"  Low bound: {low}°F")
print(f"  High bound: {high}°F")
print(f"  Means: {low} <= actual < {high}")

# Calculate for 0-day model
print(f"\n{'='*70}")
print("0-DAY MODEL (Same-day forecast)")
print(f"{'='*70}")

mean_error = error_model_0d['mean']
std_error = error_model_0d['std']

print(f"  Bias (Mean Error): {mean_error:+.2f}°F")
print(f"  Std Dev: {std_error:.2f}°F")
print(f"  MAE: {error_model_0d['mae']:.2f}°F")

# Adjust forecast for bias
adjusted_forecast = forecast_temp - mean_error

print(f"\nAdjusted Forecast:")
print(f"  Raw forecast: {forecast_temp}°F")
print(f"  Bias adjustment: {mean_error:+.2f}°F")
print(f"  Adjusted forecast: {adjusted_forecast:.2f}°F")

# Calculate z-scores
z_low = (low - adjusted_forecast) / std_error
z_high = (high - adjusted_forecast) / std_error

print(f"\nZ-scores:")
print(f"  Z-low ({low}°F): {z_low:.3f}")
print(f"  Z-high ({high}°F): {z_high:.3f}")

# Calculate probabilities
prob_below_low = stats.norm.cdf(z_low)
prob_below_high = stats.norm.cdf(z_high)
prob_in_range = prob_below_high - prob_below_low

print(f"\nProbabilities:")
print(f"  P(actual < {low}°F): {prob_below_low:.1%}")
print(f"  P(actual < {high}°F): {prob_below_high:.1%}")
print(f"  P({low} <= actual < {high}): {prob_in_range:.1%}")

print(f"\n>>> MODEL PROBABILITY (0d): {prob_in_range:.1%}")

# Calculate for 1-day model
print(f"\n{'='*70}")
print("1-DAY MODEL (1-day ahead forecast)")
print(f"{'='*70}")

mean_error_1d = error_model_1d['mean']
std_error_1d = error_model_1d['std']

print(f"  Bias (Mean Error): {mean_error_1d:+.2f}°F")
print(f"  Std Dev: {std_error_1d:.2f}°F")
print(f"  MAE: {error_model_1d['mae']:.2f}°F")

adjusted_forecast_1d = forecast_temp - mean_error_1d

print(f"\nAdjusted Forecast:")
print(f"  Raw forecast: {forecast_temp}°F")
print(f"  Bias adjustment: {mean_error_1d:+.2f}°F")
print(f"  Adjusted forecast: {adjusted_forecast_1d:.2f}°F")

z_low_1d = (low - adjusted_forecast_1d) / std_error_1d
z_high_1d = (high - adjusted_forecast_1d) / std_error_1d

print(f"\nZ-scores:")
print(f"  Z-low ({low}°F): {z_low_1d:.3f}")
print(f"  Z-high ({high}°F): {z_high_1d:.3f}")

prob_below_low_1d = stats.norm.cdf(z_low_1d)
prob_below_high_1d = stats.norm.cdf(z_high_1d)
prob_in_range_1d = prob_below_high_1d - prob_below_low_1d

print(f"\nProbabilities:")
print(f"  P(actual < {low}°F): {prob_below_low_1d:.1%}")
print(f"  P(actual < {high}°F): {prob_below_high_1d:.1%}")
print(f"  P({low} <= actual < {high}): {prob_in_range_1d:.1%}")

print(f"\n>>> MODEL PROBABILITY (1d): {prob_in_range_1d:.1%}")

# Analysis
print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}")

print(f"\nMarket Probability: 60%")
print(f"Model Probability (0d): {prob_in_range:.1%}")
print(f"Model Probability (1d): {prob_in_range_1d:.1%}")
print(f"\nEdge (0d): {(prob_in_range - 0.60):+.1%}")
print(f"Edge (1d): {(prob_in_range_1d - 0.60):+.1%}")

print(f"\nWhy the difference?")
print(f"1. Forecast is {forecast_temp}°F, range is {threshold_str}°F")
print(f"2. 0-day model has POSITIVE bias (+{mean_error:.2f}°F)")
print(f"   - Forecasts tend to be {abs(mean_error):.2f}°F too LOW")
print(f"   - So adjusted forecast is {adjusted_forecast:.2f}°F (higher)")
print(f"3. With adjusted forecast of {adjusted_forecast:.2f}°F:")
print(f"   - Range {threshold_str}°F is BELOW the adjusted forecast")
print(f"   - So probability is LOW ({prob_in_range:.1%})")
print(f"4. Market thinks 60% chance, but model thinks only {prob_in_range:.1%}")
print(f"   - Market may know something (current temps, weather patterns)")
print(f"   - Or market may be wrong (opportunity!)")

# Show distribution
print(f"\n{'='*70}")
print("DISTRIBUTION VISUALIZATION")
print(f"{'='*70}")

print(f"\nWith 0-day model:")
print(f"  Adjusted forecast: {adjusted_forecast:.1f}°F")
print(f"  Std dev: {std_error:.1f}°F")
print(f"  68% confidence interval: {adjusted_forecast - std_error:.1f}°F to {adjusted_forecast + std_error:.1f}°F")
print(f"  95% confidence interval: {adjusted_forecast - 2*std_error:.1f}°F to {adjusted_forecast + 2*std_error:.1f}°F")

print(f"\n  Target range: {low:.1f}°F to {high:.1f}°F")
print(f"  Distance from adjusted forecast: {low - adjusted_forecast:.1f}°F to {high - adjusted_forecast:.1f}°F")
print(f"  In standard deviations: {z_low:.2f}σ to {z_high:.2f}σ")

if z_low < -1:
    print(f"\n  ⚠️ Range is MORE THAN 1 std dev BELOW adjusted forecast")
    print(f"     This is why probability is low!")
elif z_low < 0:
    print(f"\n  ℹ️ Range is BELOW adjusted forecast but within 1 std dev")
else:
    print(f"\n  ✓ Range is ABOVE adjusted forecast")

print(f"\n{'='*70}")
