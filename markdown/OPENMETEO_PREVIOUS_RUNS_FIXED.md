# Open-Meteo Previous Runs API - Fixed Implementation

## Problem

The original implementation used incorrect parameter names for the Open-Meteo Previous Runs API:

- ❌ Used: `temperature_2m_day_0`, `temperature_2m_day_1`, etc.
- ✅ Correct: `temperature_2m` (day 0), `temperature_2m_previous_day1`, `temperature_2m_previous_day2`, etc.

## Solution

Created two working implementations:

### 1. **fetch_openmeteo_previous_runs.py** (New, Improved)

Location: `scripts/fetching/fetch_openmeteo_previous_runs.py`

**Features:**

- Object-oriented design with `OpenMeteoPreviousRunsFetcher` class
- Support for multiple weather variables (temperature, precipitation, wind, etc.)
- Two output formats:
  - Raw format: Wide format with all forecast runs as columns
  - Analysis format: Long format optimized for accuracy analysis
- Better error handling and retry logic
- Type hints for better code clarity

**Usage:**

```python
from scripts.fetching.fetch_openmeteo_previous_runs import OpenMeteoPreviousRunsFetcher

# Initialize fetcher
fetcher = OpenMeteoPreviousRunsFetcher(
    latitude=40.7769,
    longitude=-73.8740,
    timezone="America/New_York"
)

# Fetch for analysis (recommended for Polymarket)
df = fetcher.fetch_for_analysis(
    start_date='2025-01-01',
    end_date='2025-01-17',
    lead_times=[0, 1, 2, 3]  # Days before target date
)

# Output columns: valid_time, target_date, lead_time, forecast_issued, temperature, source
```

### 2. **fetch_previous_runs_forecasts.py** (Original, Fixed)

Location: `scripts/fetching/fetch_previous_runs_forecasts.py`

**Features:**

- Simple function-based approach
- Parallel fetching with ThreadPoolExecutor
- Automatic deduplication and CSV updates
- Progress tracking

**Usage:**

```python
from scripts.fetching.fetch_previous_runs_forecasts import update_forecasts

# Fetch and update CSV
df = update_forecasts(
    start_date='2025-01-01',
    days_before=[0, 1, 2],  # Lead times to fetch
    csv_path='data/raw/historical_forecasts.csv'
)
```

## API Parameter Reference

### Correct Variable Names

- **Day 0 (most recent):** `temperature_2m`
- **Day 1 (1 day before):** `temperature_2m_previous_day1`
- **Day 2 (2 days before):** `temperature_2m_previous_day2`
- **Day 3 (3 days before):** `temperature_2m_previous_day3`
- And so on up to `previous_day7`

### Example API Request

```
https://previous-runs-api.open-meteo.com/v1/forecast?
  latitude=40.7769&
  longitude=-73.874&
  start_date=2025-01-17&
  end_date=2025-01-17&
  hourly=temperature_2m,temperature_2m_previous_day1,temperature_2m_previous_day2&
  temperature_unit=fahrenheit&
  timezone=America/New_York
```

## Output Format

Both scripts produce data suitable for forecast accuracy analysis:

| valid_time          | target_date | lead_time | forecast_issued | temperature | source                   |
| ------------------- | ----------- | --------- | --------------- | ----------- | ------------------------ |
| 2025-01-01 00:00:00 | 2025-01-01  | 0         | 2025-01-01      | 44.9        | open_meteo_previous_runs |
| 2025-01-01 00:00:00 | 2025-01-01  | 1         | 2024-12-31      | 46.4        | open_meteo_previous_runs |
| 2025-01-01 00:00:00 | 2025-01-01  | 2         | 2024-12-30      | 51.3        | open_meteo_previous_runs |

## For Polymarket Betting

**Lead Time Interpretation:**

- `lead_time=0`: Forecast issued on the same day (nowcast) - most accurate
- `lead_time=1`: Forecast issued 1 day before - when you might place bets
- `lead_time=2`: Forecast issued 2 days before - typical market opening
- `lead_time=3`: Forecast issued 3 days before - early positioning

**Recommended Usage:**

1. Use `lead_time=2` to simulate forecast accuracy when markets typically open
2. Compare with `lead_time=1` and `lead_time=0` to see forecast evolution
3. Calculate error distributions at each lead time for uncertainty estimates

## Data Availability

- Generally available from January 2024 onwards
- Some variables (like GFS temperature) available from March 2021
- Data is continuously archived and updated

## Performance

- New implementation: ~17 dates in ~13 seconds
- Original implementation: Parallel fetching with 5 workers
- Both handle retries and rate limiting automatically
