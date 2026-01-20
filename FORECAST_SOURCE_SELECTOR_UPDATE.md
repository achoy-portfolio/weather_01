# Forecast Source Selector Update

## Overview

Added the ability to select which forecast source to use for betting recommendations. Users can now choose between Open-Meteo, NWS, Visual Crossing, or an average of all three.

## Changes Made

### 1. New Sidebar Control

**Location**: Sidebar → "Forecast Source"

**Options**:

- **Open-Meteo** (default) - Fast, reliable, always available
- **NWS** - National Weather Service official forecast
- **Visual Crossing** - Commercial weather API
- **Average of All** - Takes the average of all available forecasts

### 2. Updated Recommendation Logic

**Before**: Always used Open-Meteo forecast for recommendations

**After**: Uses the selected forecast source:

- Calculates max temperature for the betting day from selected source
- If "Average of All" is selected, averages all available forecasts
- Passes the forecast max to `generate_recommendations()` function

### 3. Visual Indicators

**Model Information Card**:

- Shows which forecast source is being used
- Displays the forecast max temperature
- Shows source name below the temperature

**Forecast Comparison Cards**:

- Selected source(s) highlighted with **green border**
- Checkmark (✓) next to selected source name
- Enhanced shadow effect on selected cards
- All sources included in average show green border

**Info Banner**:

- Shows which forecast is being used
- For "Average of All", displays calculation: "Open-Meteo (50.2°F) + NWS (49.8°F) = 50.0°F"

## How It Works

### Forecast Selection Logic

```python
if forecast_source == "Open-Meteo":
    # Use Open-Meteo max temp
    forecasted_max = openmeteo_max

elif forecast_source == "NWS":
    # Use NWS max temp
    forecasted_max = nws_max

elif forecast_source == "Visual Crossing":
    # Use Visual Crossing max temp
    forecasted_max = vc_max

elif forecast_source == "Average of All":
    # Average all available forecasts
    available = [openmeteo_max, nws_max, vc_max]
    forecasted_max = mean(available)
```

### Recommendation Generation

```python
# Old signature
generate_recommendations(forecast_df, odds_df, error_model, target_date, ...)

# New signature
generate_recommendations(forecast_max, odds_df, error_model, ...)
```

The function now receives the calculated forecast max directly instead of calculating it internally.

## Use Cases

### 1. Compare Forecast Sources

Select different sources to see how recommendations change:

- Open-Meteo might suggest betting YES on 48-49°F
- NWS might suggest betting NO on 48-49°F
- Compare edge and bet sizes across sources

### 2. Consensus Approach

Use "Average of All" to:

- Reduce forecast uncertainty
- Get more conservative recommendations
- Smooth out outlier forecasts

### 3. Trust Specific Source

If you know one source is more accurate for your region:

- Select that source exclusively
- Ignore other forecasts
- Optimize for that source's error model

## Example Scenarios

### Scenario 1: Forecasts Agree

```
Open-Meteo: 50.2°F
NWS: 49.8°F
Visual Crossing: 50.0°F
Average: 50.0°F

Result: Similar recommendations regardless of source
```

### Scenario 2: Forecasts Disagree

```
Open-Meteo: 52.0°F
NWS: 48.0°F
Visual Crossing: 50.0°F
Average: 50.0°F

Open-Meteo recommendations:
- YES on 52-53°F (edge: +15%)
- NO on 48-49°F (edge: +20%)

NWS recommendations:
- YES on 48-49°F (edge: +12%)
- NO on 52-53°F (edge: +18%)

Average recommendations:
- YES on 50-51°F (edge: +10%)
- NO on 46-47°F (edge: +15%)
- NO on 54-55°F (edge: +15%)
```

### Scenario 3: Missing Forecasts

```
Open-Meteo: 50.0°F
NWS: Not available
Visual Crossing: Not available

"Average of All" uses only Open-Meteo (same as selecting Open-Meteo)
```

## Visual Guide

### Selected Source (Green Border)

```
┌─────────────────────────────────┐
│ Open-Meteo Max ✓                │ ← Green border + checkmark
│                                 │
│        50.2°F                   │
│                                 │
│ For Jan 21                      │
└─────────────────────────────────┘
```

### Not Selected (Normal)

```
┌─────────────────────────────────┐
│ NWS Max                         │ ← Normal border
│                                 │
│        49.8°F                   │
│                                 │
│ For Jan 21                      │
└─────────────────────────────────┘
```

### Average of All (Multiple Green Borders)

```
┌─────────────────────────────────┐
│ Open-Meteo Max                  │ ← Green border
│        50.2°F                   │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ NWS Max                         │ ← Green border
│        49.8°F                   │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ Visual Crossing Max             │ ← Green border
│        50.0°F                   │
└─────────────────────────────────┘

Info: Using average of: Open-Meteo (50.2°F), NWS (49.8°F), Visual Crossing (50.0°F) = 50.0°F
```

## Benefits

1. **Flexibility**: Choose the forecast source you trust most
2. **Comparison**: See how different forecasts affect recommendations
3. **Risk Management**: Use average to reduce forecast uncertainty
4. **Transparency**: Clear visual indication of which source is being used
5. **Adaptability**: Can switch sources if one becomes unavailable

## Technical Details

### Function Signature Change

**Before**:

```python
def generate_recommendations(forecast_df, odds_df, error_model, target_date, ...):
    # Calculate forecast max internally
    betting_day_forecast = forecast_df[forecast_df['date'] == target_date]
    forecasted_max = betting_day_forecast['temp_f'].max()
    # ... rest of logic
```

**After**:

```python
def generate_recommendations(forecast_max, odds_df, error_model, ...):
    # Forecast max is passed in directly
    # ... rest of logic uses forecast_max
```

### Calculation Logic

```python
# Calculate forecasted max based on selected source
forecasted_max_betting_day = None

if forecast_source == "Open-Meteo" and forecast_df is not None:
    betting_day_data = forecast_df[forecast_df['date'] == selected_date]
    if not betting_day_data.empty:
        forecasted_max_betting_day = betting_day_data['temp_f'].max()

elif forecast_source == "NWS" and nws_forecast_df is not None:
    betting_day_data = nws_forecast_df[nws_forecast_df['date'] == selected_date]
    if not betting_day_data.empty:
        forecasted_max_betting_day = betting_day_data['temp_f'].max()

elif forecast_source == "Visual Crossing" and vc_forecast_df is not None:
    betting_day_data = vc_forecast_df[vc_forecast_df['date'] == selected_date]
    if not betting_day_data.empty:
        forecasted_max_betting_day = betting_day_data['temp_f'].max()

elif forecast_source == "Average of All":
    max_temps = []
    # Collect all available forecasts
    if forecast_df is not None:
        betting_day_data = forecast_df[forecast_df['date'] == selected_date]
        if not betting_day_data.empty:
            max_temps.append(betting_day_data['temp_f'].max())
    # ... repeat for NWS and Visual Crossing

    if max_temps:
        forecasted_max_betting_day = sum(max_temps) / len(max_temps)
```

## Best Practices

1. **Start with Open-Meteo**: It's the default and most reliable
2. **Compare sources**: Check if they agree before placing bets
3. **Use average for uncertainty**: When forecasts disagree significantly
4. **Trust local knowledge**: If you know one source is better for your area
5. **Monitor accuracy**: Track which source performs best over time

## Future Enhancements

Potential improvements:

1. **Weighted average**: Give more weight to more accurate sources
2. **Source-specific error models**: Different error models for each source
3. **Confidence intervals**: Show uncertainty range for each source
4. **Historical accuracy**: Display past performance of each source
5. **Auto-select best**: Automatically choose most accurate source

## Troubleshooting

### No recommendations shown

- Check if selected forecast source has data for the betting day
- Try "Average of All" to use any available forecast
- Verify the date is within forecast range (tomorrow to 16 days ahead)

### Recommendations change drastically

- Normal if forecasts disagree significantly
- Review each forecast source individually
- Consider using "Average of All" for more stable recommendations

### One source always unavailable

- NWS: May not have data for far future dates
- Visual Crossing: Requires API key in .env file
- Open-Meteo: Should always be available (most reliable)
