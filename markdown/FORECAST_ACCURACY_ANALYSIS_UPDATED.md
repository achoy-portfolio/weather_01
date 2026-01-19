# Forecast Accuracy Analysis - Updated

## Changes Made

Updated `scripts/analysis/forecast_accuracy_analysis.py` to work with the new Open-Meteo Previous Runs CSV format.

### Key Updates

1. **Data Loading**
   - Now supports both `openmeteo_previous_runs.csv` and `historical_forecasts.csv`
   - Automatically detects which forecast file is available
   - Properly parses `valid_time`, `forecast_issued`, and `lead_time` columns

2. **Metric 1: Hourly Temperature Accuracy by Lead Time**
   - Analyzes hourly forecast accuracy for each lead time (0, 1, 2, 3 days)
   - Shows how accuracy degrades as lead time increases
   - Useful for understanding general forecast skill

3. **Metric 2: Daily Maximum Temperature Accuracy by Lead Time**
   - **This is the KEY metric for Polymarket betting**
   - Calculates forecasted daily max vs actual daily max for each lead time
   - Shows accuracy when market opens (2 days before) vs day before event
   - Provides actionable insights for betting decisions

### CSV Format Expected

The script expects forecast data in this format:

```csv
valid_time,target_date,lead_time,forecast_issued,temperature,source
2025-01-01 00:00:00,2025-01-01,0,2025-01-01,44.9,open_meteo_previous_runs
2025-01-01 00:00:00,2025-01-01,1,2024-12-31,46.4,open_meteo_previous_runs
2025-01-01 00:00:00,2025-01-01,2,2024-12-30,51.3,open_meteo_previous_runs
```

**Columns:**

- `valid_time`: When the forecast is valid for (datetime)
- `target_date`: Date being forecasted (date)
- `lead_time`: Days before target date (0, 1, 2, 3, etc.)
- `forecast_issued`: When forecast was issued (datetime)
- `temperature`: Forecasted temperature (°F)
- `source`: Data source identifier

### Output

The script produces:

1. **Console output** with detailed accuracy metrics
2. **JSON file** at `data/processed/forecast_accuracy_metrics.json` with:
   ```json
   {
     "created_at": "2026-01-18T22:46:17.176000",
     "metric_1_hourly_accuracy": {
       "overall": {...},
       "by_lead_time": {
         "0d": {"mae": 2.94, "rmse": 4.19, "bias": -2.46, "count": 19},
         "1d": {"mae": 4.92, "rmse": 6.24, "bias": -4.89, "count": 19},
         "2d": {"mae": 4.96, "rmse": 5.64, "bias": -4.96, "count": 19},
         "3d": {"mae": 5.15, "rmse": 5.81, "bias": -5.15, "count": 19}
       }
     },
     "metric_2_daily_max_accuracy": {
       "overall": {...},
       "by_lead_time": {
         "0d": {"mae": 0.20, "rmse": 0.20, "bias": -0.20, "count": 1},
         "1d": {"mae": 2.90, "rmse": 2.90, "bias": -2.90, "count": 1},
         "2d": {"mae": 5.20, "rmse": 5.20, "bias": -5.20, "count": 1},
         "3d": {"mae": 5.60, "rmse": 5.60, "bias": -5.60, "count": 1}
       }
     }
   }
   ```

### Example Output

```
======================================================================
METRIC 2: DAILY MAXIMUM TEMPERATURE ACCURACY BY LEAD TIME
======================================================================
Using Weather Underground daily max temperatures (official Polymarket source)
Matched 4 daily forecast-actual pairs

Overall Daily Max Accuracy:
  MAE:   3.48°F
  RMSE:  4.09°F
  Bias:  -3.48°F
  Count: 4 days

Accuracy by Lead Time (days before):
Lead Time    MAE      RMSE     Bias     Count
--------------------------------------------------
0 day(s)        0.20°F    0.20°F   -0.20°F       1 (same day - nowcast)
1 day(s)        2.90°F    2.90°F   -2.90°F       1 (day before event)
2 day(s)        5.20°F    5.20°F   -5.20°F       1 (market opens)
3 day(s)        5.60°F    5.60°F   -5.60°F       1

Sample predictions (lead_time=2, market opening):
target_date  forecasted_max  max_temp_f  error
 2025-01-17            37.8        43.0   -5.2
```

### Interpretation for Polymarket Betting

**When market opens (2 days before):**

- Expect ±5.2°F error on average
- Forecasts tend to be slightly cold (bias: -5.2°F)
- If forecast says 45°F, actual will likely be between 40-50°F

**Day before event:**

- Accuracy improves to ±2.9°F
- Consider updating positions if forecast changes significantly

**Same day (too late for betting):**

- Very accurate (±0.2°F) but markets are closed

### Usage

```bash
# Run the analysis
python scripts/analysis/forecast_accuracy_analysis.py

# Output will be saved to:
# data/processed/forecast_accuracy_metrics.json
```

### Requirements

**Forecast Data:**

- `data/raw/openmeteo_previous_runs.csv` (preferred)
- OR `data/raw/historical_forecasts.csv`

**Actual Temperature Data:**

- `data/raw/wunderground_hourly_temps.csv` (preferred - official Polymarket source)
- OR `data/raw/actual_temperatures_meteo.csv` (fallback)

**Optional:**

- `data/raw/wunderground_daily_max_temps.csv` (for more accurate daily max comparisons)

### Next Steps

To get more robust accuracy metrics:

1. **Fetch more historical forecast data:**

   ```bash
   python scripts/fetching/fetch_openmeteo_previous_runs.py
   # Or
   python scripts/fetching/fetch_previous_runs_forecasts.py
   ```

2. **Fetch more actual temperature data:**

   ```bash
   python scripts/fetching/fetch_wunderground_actual.py
   ```

3. **Re-run the analysis:**
   ```bash
   python scripts/analysis/forecast_accuracy_analysis.py
   ```

With more data (e.g., 30+ days), the accuracy metrics will be more reliable and useful for betting decisions.
