"""
Fetch historical weather forecasts from Open-Meteo Previous Runs API.

This API provides forecasts issued at different lead times (Day 0, Day 1, Day 2, etc.)
allowing us to analyze how forecast accuracy changes as we get closer to the event.

Perfect for Polymarket betting analysis!

API Documentation: https://open-meteo.com/en/docs/previous-runs-api
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time as time_module
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# KLGA coordinates
KLGA_LAT = 40.7769
KLGA_LON = -73.8740

# API configuration
API_BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
TIMEOUT = 30
MAX_WORKERS = 5
REQUEST_DELAY = 0.5


def fetch_previous_runs_for_date(
    target_date,
    days_before=[0, 1, 2, 3],
    lat=KLGA_LAT,
    lon=KLGA_LON,
    retry_count=0
):
    """
    Fetch forecasts for a target date from different days before.
    
    Args:
        target_date: Date to get forecasts for (datetime.date or string 'YYYY-MM-DD')
        days_before: List of days before to fetch (e.g., [0, 1, 2] for Day 0, Day 1, Day 2)
        lat: Latitude (default: KLGA)
        lon: Longitude (default: KLGA)
        retry_count: Current retry attempt
    
    Returns:
        list: List of forecast records with keys:
            - forecast_date: Date when forecast was issued (target_date - days_before)
            - forecast_time: Time when forecast was issued (always 21:00 for consistency)
            - days_before: How many days before the target date
            - valid_time: Target datetime
            - temperature: Forecasted temperature (°F)
            - source: Data source identifier
        None: If fetch fails
    """
    # Convert target_date to datetime.date if string
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    
    # Build temperature variable list for each day
    # Day 0 is just 'temperature_2m', previous days use '_previous_day1', '_previous_day2', etc.
    temperature_vars = []
    for day in days_before:
        if day == 0:
            temperature_vars.append('temperature_2m')
        else:
            temperature_vars.append(f'temperature_2m_previous_day{day}')
    
    # API parameters
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': target_date.strftime('%Y-%m-%d'),
        'end_date': target_date.strftime('%Y-%m-%d'),
        'hourly': ','.join(temperature_vars),
        'temperature_unit': 'fahrenheit',
        'timezone': 'America/New_York'
    }
    
    try:
        time_module.sleep(REQUEST_DELAY)
        
        response = requests.get(API_BASE_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        if 'hourly' not in data:
            logger.warning(f"No hourly data for {target_date}")
            return None
        
        hourly = data['hourly']
        times = hourly['time']
        
        results = []
        
        # Process each day_before
        for day in days_before:
            if day == 0:
                var_name = 'temperature_2m'
            else:
                var_name = f'temperature_2m_previous_day{day}'
            
            if var_name not in hourly:
                logger.warning(f"No data for {var_name} on {target_date}")
                continue
            
            temps = hourly[var_name]
            
            # Calculate forecast issue date (target_date - days_before)
            forecast_date = target_date - timedelta(days=day)
            
            # For each hour in the target date
            for time_str, temp in zip(times, temps):
                if temp is None:
                    continue
                
                valid_dt = datetime.fromisoformat(time_str)
                
                results.append({
                    'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                    'forecast_time': '21:00',  # Standardize to 9 PM for consistency
                    'days_before': day,
                    'valid_time': valid_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'temperature': temp,
                    'source': 'open_meteo_previous_runs'
                })
        
        if results:
            logger.info(f"✓ Fetched {len(results)} forecasts for {target_date} from {len(days_before)} different lead times")
            return results
        else:
            logger.warning(f"No forecast data for {target_date}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching forecasts for {target_date}")
        return _retry_with_backoff(fetch_previous_runs_for_date, target_date, days_before, lat, lon, retry_count)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API error for {target_date}: {e}")
        return _retry_with_backoff(fetch_previous_runs_for_date, target_date, days_before, lat, lon, retry_count)
        
    except Exception as e:
        logger.error(f"Error parsing response for {target_date}: {e}")
        return None


def _retry_with_backoff(func, target_date, days_before, lat, lon, retry_count):
    """Retry with exponential backoff."""
    if retry_count >= MAX_RETRIES:
        logger.error(f"Max retries ({MAX_RETRIES}) exceeded for {target_date}")
        return None
    
    delay = INITIAL_RETRY_DELAY * (2 ** retry_count)
    logger.info(f"Retrying in {delay} seconds... (attempt {retry_count + 1}/{MAX_RETRIES})")
    time_module.sleep(delay)
    
    return func(target_date, days_before, lat, lon, retry_count + 1)


def fetch_previous_runs_batch(
    start_date,
    end_date=None,
    days_before=[0, 1, 2, 3],
    lat=KLGA_LAT,
    lon=KLGA_LON
):
    """
    Fetch previous run forecasts for a range of dates.
    
    Args:
        start_date: First target date (datetime.date or string 'YYYY-MM-DD')
        end_date: Last target date (default: yesterday)
        days_before: List of days before to fetch (default: [0, 1, 2, 3])
        lat: Latitude (default: KLGA)
        lon: Longitude (default: KLGA)
    
    Returns:
        pd.DataFrame: DataFrame with forecast data
    """
    # Convert dates
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    
    if end_date is None:
        end_date = datetime.now().date() - timedelta(days=1)
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    logger.info(f"Fetching forecasts from {start_date} to {end_date}")
    logger.info(f"Lead times: {days_before} days before each target date")
    
    # Generate list of target dates
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    
    logger.info(f"Total target dates: {len(dates)}")
    
    # Fetch forecasts in parallel
    all_forecasts = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_date = {
            executor.submit(fetch_previous_runs_for_date, date, days_before, lat, lon): date
            for date in dates
        }
        
        # Collect results
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            completed += 1
            
            try:
                result = future.result()
                if result:
                    all_forecasts.extend(result)
                
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{len(dates)} dates completed")
                    
            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
    
    # Convert to DataFrame
    if all_forecasts:
        df = pd.DataFrame(all_forecasts)
        logger.info(f"✓ Successfully fetched {len(df)} forecast records")
        return df
    else:
        logger.warning("No forecasts retrieved")
        return pd.DataFrame(columns=[
            'forecast_date', 'forecast_time', 'days_before',
            'valid_time', 'temperature', 'source'
        ])


def save_forecasts_to_csv(df, output_path='data/raw/historical_forecasts.csv'):
    """Save forecasts DataFrame to CSV file."""
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved {len(df)} forecasts to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")
        return False


def load_existing_forecasts(csv_path='data/raw/historical_forecasts.csv'):
    """Load existing forecasts from CSV file."""
    try:
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} existing forecast records from {csv_path}")
            return df
        else:
            logger.info(f"No existing forecast file found at {csv_path}")
            return pd.DataFrame(columns=[
                'forecast_date', 'forecast_time', 'days_before',
                'valid_time', 'temperature', 'source'
            ])
    except Exception as e:
        logger.error(f"Error loading existing forecasts: {e}")
        return pd.DataFrame(columns=[
            'forecast_date', 'forecast_time', 'days_before',
            'valid_time', 'temperature', 'source'
        ])


def update_forecasts(
    start_date='2025-01-01',
    end_date=None,
    days_before=[0, 1, 2],
    csv_path='data/raw/historical_forecasts.csv'
):
    """
    Update forecast CSV with new data, avoiding duplicates.
    
    Args:
        start_date: First target date to fetch (default: 2025-01-01)
        end_date: Last target date to fetch (default: yesterday)
        days_before: List of days before to fetch (default: [0, 1, 2])
        csv_path: Path to CSV file
    
    Returns:
        pd.DataFrame: Updated forecasts DataFrame
    """
    # Load existing forecasts
    existing_df = load_existing_forecasts(csv_path)
    
    # Determine which dates need to be fetched
    if len(existing_df) > 0:
        existing_df['valid_time_parsed'] = pd.to_datetime(existing_df['valid_time'])
        existing_dates = set(existing_df['valid_time_parsed'].dt.date.unique())
        logger.info(f"Found {len(existing_dates)} existing target dates")
    else:
        existing_dates = set()
    
    # Convert start/end dates
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if end_date is None:
        end_date = datetime.now().date() - timedelta(days=1)
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Filter to only dates we don't have yet
    dates_to_fetch = []
    current = start_date
    while current <= end_date:
        if current not in existing_dates:
            dates_to_fetch.append(current)
        current += timedelta(days=1)
    
    if not dates_to_fetch:
        logger.info("No new dates to fetch (all already exist)")
        return existing_df
    
    logger.info(f"Need to fetch forecasts for {len(dates_to_fetch)} new dates")
    
    # Fetch new forecasts
    new_df = fetch_previous_runs_batch(
        dates_to_fetch[0],
        dates_to_fetch[-1] if len(dates_to_fetch) > 1 else dates_to_fetch[0],
        days_before=days_before
    )
    
    if len(new_df) > 0:
        logger.info(f"Adding {len(new_df)} new forecast records")
        
        # Combine with existing
        if len(existing_df) > 0:
            if 'valid_time_parsed' in existing_df.columns:
                existing_df = existing_df.drop(columns=['valid_time_parsed'])
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(
            subset=['forecast_date', 'days_before', 'valid_time'],
            keep='first'
        )
        
        # Sort
        combined_df['forecast_date_parsed'] = pd.to_datetime(combined_df['forecast_date'])
        combined_df['valid_time_parsed'] = pd.to_datetime(combined_df['valid_time'])
        combined_df = combined_df.sort_values(['valid_time_parsed', 'days_before'])
        combined_df = combined_df.drop(columns=['forecast_date_parsed', 'valid_time_parsed'])
        
        # Save
        save_forecasts_to_csv(combined_df, csv_path)
        
        return combined_df
    else:
        logger.warning("No new forecasts fetched")
        return existing_df


if __name__ == "__main__":
    print("=" * 70)
    print("Previous Runs Forecast Fetcher")
    print("=" * 70)
    print("Fetching Open-Meteo Previous Runs forecasts for KLGA")
    print("This API provides forecasts issued at different lead times")
    print("Perfect for analyzing forecast accuracy for Polymarket betting!")
    print()
    
    # Note: Previous Runs API data is available from January 2024 onwards
    # Starting from 2025 as requested
    df = update_forecasts(
        start_date='2025-01-01',  # Start from 2025
        days_before=[0, 1, 2]  # Day 0 (nowcast), Day 1 (1 day ago), Day 2 (2 days ago)
    )
    
    if len(df) > 0:
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Total forecast records: {len(df)}")
        
        # Count unique dates and lead times
        unique_targets = df['valid_time'].nunique()
        unique_lead_times = df['days_before'].nunique()
        print(f"Unique target dates: {unique_targets}")
        print(f"Unique lead times: {unique_lead_times}")
        
        # Show lead time distribution
        print(f"\nForecasts by lead time:")
        print(df['days_before'].value_counts().sort_index())
        
        # Show sample
        print(f"\nSample forecasts:")
        sample = df.head(20)
        print(sample[['forecast_date', 'days_before', 'valid_time', 'temperature']].to_string())
        
        print(f"\nSaved to: data/raw/historical_forecasts.csv")
        print("=" * 70)
        
        print("\n" + "=" * 70)
        print("WHAT THIS DATA MEANS:")
        print("=" * 70)
        print("- days_before=0: Forecast issued on the target date (nowcast)")
        print("- days_before=1: Forecast issued 1 day before target date")
        print("- days_before=2: Forecast issued 2 days before target date")
        print("- days_before=3: Forecast issued 3 days before target date")
        print("\nFor Polymarket betting:")
        print("- Use days_before=2 to see forecast accuracy when markets open")
        print("- Compare with days_before=1 to see if waiting helps")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("No forecasts available")
        print("=" * 70)
