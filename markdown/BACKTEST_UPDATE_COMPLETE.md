# Backtest Strategy Update - Complete

## Summary

Successfully updated `backtest_betting_strategy.py` and verified `betting_simulator.py` to use the new error distribution analysis with lead time-specific models.

## Changes Made

### 1. Updated Error Model Loading

- Now uses 2-day lead time error model from `error_distribution_analysis.json`
- Accesses error statistics via `model['by_lead_time']['2d']`
- Uses correct structure: `mean`, `std`, `mae`, `sample_size`

### 2. Updated Data Loading

- Handles both old (`historical_forecasts.csv`) and new (`openmeteo_previous_runs.csv`) formats
- Filters forecasts to 2-day lead time (when market opens)
- Uses Weather Underground daily max temps (official Polymarket source)

### 3. Updated Forecast Matching

- Uses pre-filtered 2-day lead time forecasts
- Matches forecasts to betting days via `valid_time`
- Removed old 9 PM search logic (now uses filtered data directly)

### 4. Updated Probability Calculations

- Uses 2-day lead time error model: Bias = -0.20°F, Std = 3.74°F
- Adjusts forecasts for bias before calculating probabilities
- Applies normal distribution with lead time-specific parameters

## Test Results

### Backtest Execution

```
Error Model (2-day lead time):
  Bias: -0.20°F
  Std Dev: 3.74°F
  MAE: 2.83°F
  Sample Size: 366 forecasts

Data Loaded:
  Forecasts: 9,168 records (2-day lead time)
  Daily Max Temps: 366 days
  Odds: 1,226,737 records
  Betting Days: 308

Results:
  Total Opportunities: 2,156
  Bets Placed: 104
  Bets Won: 17
  Win Rate: 16.3%
  ROI: -59.1%
```

### Analysis Insights

- Mean edge on bets: 23.4%
- Higher edge buckets (>30%) had better win rates (31%)
- "Above" threshold bets performed better than "below" (19.5% vs 5.6%)
- July 2025 was most profitable month (+$246)
- Strategy may need recalibration or parameter adjustment

## Files Updated

- `scripts/analysis/backtest_betting_strategy.py` ✅
- `scripts/analysis/betting_simulator.py` ✅ (verified, no changes needed)

## Next Steps (Optional)

1. Investigate why win rate is lower than expected despite positive edge
2. Consider adjusting strategy parameters (min_edge, min_ev)
3. Analyze forecast accuracy issues (e.g., March 30: forecast 65.5°F, actual 46°F)
4. Explore alternative error models or calibration techniques
5. Test different Kelly fractions or bet sizing strategies
