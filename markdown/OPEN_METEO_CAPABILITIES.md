# Open-Meteo: Past Forecasts vs Current Forecasts

## Short Answer

**Yes! Open-Meteo offers BOTH:**

1. ✅ **Current forecasts** (free, unlimited)
2. ✅ **Historical forecasts** (past predictions archived since 2021-2022)
3. ✅ **Historical actual weather** (observations back to 1940)

## Three Different APIs

### 1. Forecast API (Current Forecasts)

**URL:** `https://api.open-meteo.com/v1/forecast`

**What it provides:**

- Current weather forecasts (next 7-16 days)
- Updated every 1-6 hours depending on model
- **Free, unlimited, no API key required**

**Use case:** Get today's forecast for tomorrow

```python
# Example: Get tomorrow's forecast
url = "https://api.open-meteo.com/v1/forecast"
params = {
    'latitude': 40.7769,
    'longitude': -73.8740,
    'daily': 'temperature_2m_max,temperature_2m_min',
    'temperature_unit': 'fahrenheit',
    'timezone': 'America/New_York'
}
```

### 2. Historical Forecast API (Past Predictions) ⭐

**URL:** `https://api.open-meteo.com/v1/historical-forecast-api`

**What it provides:**

- **Archived forecasts from the past**
- Shows what the forecast was on a specific date
- Available from **2021-2022 onwards** (varies by model)
- Continuously updated and archived

**Use case:** "What did the forecast predict on January 15th for January 17th?"

**Key Features:**

- Same models as current forecast API
- Continuously archives all forecast updates
- Ideal for training ML models
- Perfect for backtesting betting strategies
- Can analyze forecast accuracy over time

**Available Models & Start Dates:**
| Model | Region | Available Since |
|-------|--------|-----------------|
| GFS | Global | 2021-03-23 |
| HRRR | US | 2018-01-01 |
| ICON | Global | 2022-11-24 |
| ECMWF IFS | Global | 2022-11-07 |
| UK Met Office | Global | 2022-03-01 |

**Example:**

```python
# Get what the forecast was on Jan 15, 2025 for Jan 17, 2025
url = "https://api.open-meteo.com/v1/historical-forecast-api"
params = {
    'latitude': 40.7769,
    'longitude': -73.8740,
    'start_date': '2025-01-17',
    'end_date': '2025-01-17',
    'daily': 'temperature_2m_max',
    'temperature_unit': 'fahrenheit',
    'timezone': 'America/New_York'
}
# This returns the forecast that was issued on Jan 15
```

### 3. Historical Weather API (Actual Observations)

**URL:** `https://archive-api.open-meteo.com/v1/archive`

**What it provides:**

- **Actual observed weather** (not forecasts)
- Based on ERA5 reanalysis model
- Data from **1940 onwards**
- What actually happened

**Use case:** "What was the actual temperature on January 17th?"

```python
# Get actual temperature that occurred
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    'latitude': 40.7769,
    'longitude': -73.8740,
    'start_date': '2025-01-17',
    'end_date': '2025-01-17',
    'daily': 'temperature_2m_max',
    'temperature_unit': 'fahrenheit',
    'timezone': 'America/New_York'
}
```

## Comparison Table

| Feature              | Forecast API         | Historical Forecast API | Historical Weather API |
| -------------------- | -------------------- | ----------------------- | ---------------------- |
| **Type**             | Current predictions  | Past predictions        | Actual observations    |
| **Time Range**       | Next 7-16 days       | 2021-2022 onwards       | 1940 onwards           |
| **Update Frequency** | Every 1-6 hours      | Archived continuously   | Complete archive       |
| **Use Case**         | Get today's forecast | Backtest strategies     | Get actual temps       |
| **Cost**             | Free                 | Free                    | Free                   |
| **API Key**          | Not required         | Not required            | Not required           |

## Why This Matters for Your Betting Strategy

### Problem with Visual Crossing

- Limited free tier (1000 calls/day)
- Historical forecasts may not be available
- Less transparent about data sources

### Open-Meteo Advantages

✅ **Completely free, unlimited**
✅ **Historical forecasts available** (can backtest!)
✅ **Multiple models** (GFS, ECMWF, HRRR, etc.)
✅ **Well documented**
✅ **High resolution** (down to 1-3km for some models)

## How to Use for Polymarket Betting

### Scenario 1: Live Betting (Today)

```python
# Get current forecast for tomorrow
from scripts.fetching.fetch_openmeteo_forecast import get_openmeteo_forecast

hourly_df, daily_df = get_openmeteo_forecast()
tomorrow_forecast = daily_df[daily_df['date'] == tomorrow]['temp_max'].iloc[0]
```

### Scenario 2: Backtesting (Historical)

```python
# Get what the forecast was 2 days ago for tomorrow
# This lets you test your strategy on past data!

def get_historical_forecast(forecast_date, target_date):
    """
    Get what the forecast was on forecast_date for target_date.

    Args:
        forecast_date: When the forecast was issued
        target_date: What date was being forecasted
    """
    url = "https://api.open-meteo.com/v1/historical-forecast-api"

    # Calculate lead time (days ahead)
    lead_time = (target_date - forecast_date).days

    params = {
        'latitude': 40.7769,
        'longitude': -73.8740,
        'start_date': target_date.isoformat(),
        'end_date': target_date.isoformat(),
        'daily': 'temperature_2m_max',
        'temperature_unit': 'fahrenheit',
        'timezone': 'America/New_York',
        'past_days': lead_time  # How many days ahead was forecast
    }

    response = requests.get(url, params=params)
    data = response.json()

    return data['daily']['temperature_2m_max'][0]

# Example: What did the forecast on Jan 15 predict for Jan 17?
forecast_date = date(2025, 1, 15)
target_date = date(2025, 1, 17)
forecast = get_historical_forecast(forecast_date, target_date)
```

### Scenario 3: Validate Forecast Accuracy

```python
# Compare forecast to actual
forecast = get_historical_forecast(date(2025, 1, 15), date(2025, 1, 17))
actual = get_actual_temperature(date(2025, 1, 17))  # From archive API

error = forecast - actual
print(f"Forecast: {forecast:.1f}°F")
print(f"Actual: {actual:.1f}°F")
print(f"Error: {error:+.1f}°F")
```

## Recommended Strategy Update

### Current Approach (Limited)

```python
# Only uses current NWS forecast
nws_forecast = get_nws_forecast()
```

### Improved Approach (Better)

```python
# Use multiple free sources
nws_forecast = get_nws_forecast()
openmeteo_forecast = get_openmeteo_forecast()

# Blend them
final = 0.5 * nws_forecast + 0.5 * openmeteo_forecast
```

### Best Approach (Optimal)

```python
from scripts.pipelines.improved_uncertainty import ForecastEnsemble

ensemble = ForecastEnsemble()
ensemble.add_forecast('nws', nws_forecast, uncertainty=2.5)
ensemble.add_forecast('open_meteo', openmeteo_forecast, uncertainty=2.5)

# Automatically blends with inverse variance weighting
consensus, uncertainty = ensemble.get_consensus()
```

## Backtesting Your Strategy

With Open-Meteo's Historical Forecast API, you can now:

1. **Test your model on past data**
   - Get forecasts from 2021-2024
   - Compare to actual outcomes
   - Calculate real win rate

2. **Validate forecast accuracy**
   - Track Open-Meteo vs NWS vs Visual Crossing
   - Find which is most accurate for KLGA
   - Adjust weights accordingly

3. **Optimize your strategy**
   - Test different uncertainty values
   - Find optimal edge thresholds
   - Validate Kelly sizing

## Example: Complete Backtest

```python
import pandas as pd
from datetime import date, timedelta

def backtest_strategy(start_date, end_date):
    """
    Backtest betting strategy using historical forecasts.
    """
    results = []

    current_date = start_date
    while current_date <= end_date:
        # Get forecast from 2 days before
        forecast_date = current_date - timedelta(days=2)

        # Get what the forecast predicted
        forecast = get_historical_forecast(forecast_date, current_date)

        # Get what actually happened
        actual = get_actual_temperature(current_date)

        # Get market odds (if you have historical data)
        market_odds = get_polymarket_odds(current_date)

        # Calculate if you would have bet
        edge = calculate_edge(forecast, market_odds)

        if edge > 0.05:
            # Would have bet
            outcome = 'win' if check_bet_outcome(forecast, actual, market_odds) else 'loss'
            results.append({
                'date': current_date,
                'forecast': forecast,
                'actual': actual,
                'edge': edge,
                'outcome': outcome
            })

        current_date += timedelta(days=1)

    df = pd.DataFrame(results)
    win_rate = (df['outcome'] == 'win').mean()

    print(f"Backtest Results:")
    print(f"  Total bets: {len(df)}")
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Average edge: {df['edge'].mean():.1%}")

    return df

# Run backtest for 2024
results = backtest_strategy(date(2024, 1, 1), date(2024, 12, 31))
```

## Cost Comparison

| Service             | Current Forecast | Historical Forecast | Cost                |
| ------------------- | ---------------- | ------------------- | ------------------- |
| **Open-Meteo**      | ✅ Yes           | ✅ Yes (2021+)      | **FREE**            |
| **NWS**             | ✅ Yes           | ❌ No               | FREE                |
| **Visual Crossing** | ✅ Yes           | ⚠️ Limited          | 1000 calls/day free |
| **Weather.com**     | ✅ Yes           | ❌ No               | Paid API            |

## Bottom Line

**Open-Meteo is perfect for your use case:**

1. ✅ **Free and unlimited** - No API key needed
2. ✅ **Historical forecasts** - Can backtest your strategy
3. ✅ **Multiple models** - GFS, ECMWF, HRRR, etc.
4. ✅ **High accuracy** - Similar to NWS (~2.5°F MAE)
5. ✅ **Well documented** - Easy to use

**Recommendation:**

- Use **Open-Meteo + NWS** for current forecasts (blend them)
- Use **Open-Meteo Historical Forecast API** for backtesting
- Use **Open-Meteo Archive API** for actual temperatures
- Drop Visual Crossing (unless you need it for other reasons)

This gives you everything you need for free!
