"""
Fetch historical weather forecasts from Open-Meteo Historical Forecast API.

This script retrieves archived forecasts that were issued at specific times in the past,
allowing us to analyze forecast accuracy and build error models for betting strategies.

API Documentation: https://open-meteo.com/en/docs/historical-forecast-api
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, time
from pathlib import Path
import time as time_module
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
API_BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1  # seconds
TIMEOUT = 10  # seconds
MAX_REQUESTS_PER_MINUTE = 600
REQUEST_DELAY = 60.0 / MAX_REQUESTS_PER_MINUTE  # ~0.1 seconds between requests

# Thread-safe rate limiter
class RateLimiter:
    def __init__(self, max_per_minute):
        self.max_per_minute = max_per_minute
        self.delay = 60.0 / max_per_minute
        self.last_request_time = 0
        self.lock = threading.Lock()
    
    def wait(self):
        with self.lock:
            current_time = time_module.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.delay:
                sleep_time = self.delay - time_since_last
                time_module.sleep(sleep_time)
            self.last_request_time = time_module.time()

rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)


def fetch_historical_forecast_for_date(
    forecast_date,
    forecast_time_hour=21,
    lat=KLGA_LAT,
    lon=KLGA_LON,
    retry_count=0
):
    """
    Fetch historical hourly forecast for 3 days ahead from a specific forecast issue date/time.
    
    The forecast is retrieved as it was issued at a specific time (default 9 PM / 21:00)
    and includes hourly forecasts for the next 72 hours (3 days).
    
    Args:
        forecast_date: Date when forecast was issued (datetime.date or string 'YYYY-MM-DD')
        forecast_time_hour: Hour when forecast was issued (0-23, default 21 for 9 PM)
        lat: Latitude (default: KLGA)
        lon: Longitude (default: KLGA)
        retry_count: Current retry attempt (used internally)
    
    Returns:
        list: List of forecast records, one per hour for 72 hours, each with keys:
            - forecast_date: Date when forecast was issued
            - forecast_time: Time when forecast was issued (HH:00)
            - forecast_hour: Hour offset from forecast issue time (0-71)
            - valid_time: Datetime when this forecast is valid for
            - temperature: Forecasted temperature (°F)
            - source: Data source identifier
        None: If fetch fails after all retries
    """
    # Convert forecast_date to datetime.date if string
    if isinstance(forecast_date, str):
        forecast_date = datetime.strptime(forecast_date, '%Y-%m-%d').date()
    
    # Calculate date range for 3 days ahead
    start_date = forecast_date
    end_date = forecast_date + timedelta(days=3)
    
    # Build API parameters for hourly data
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'hourly': 'temperature_2m',
        'temperature_unit': 'fahrenheit',
        'timezone': 'America/New_York'
    }
    
    try:
        # Apply rate limiting
        rate_limiter.wait()
        
        response = requests.get(API_BASE_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract hourly forecast data
        if 'hourly' in data and 'temperature_2m' in data['hourly']:
            hourly_times = data['hourly']['time']
            hourly_temps = data['hourly']['temperature_2m']
            
            # Create forecast issue datetime
            forecast_issue_dt = datetime.combine(forecast_date, time(forecast_time_hour, 0))
            
            results = []
            for i, (time_str, temp) in enumerate(zip(hourly_times, hourly_temps)):
                valid_dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                
                # Only include forecasts from the issue time onwards for 72 hours
                if valid_dt >= forecast_issue_dt:
                    hours_ahead = int((valid_dt - forecast_issue_dt).total_seconds() / 3600)
                    
                    if hours_ahead < 72:  # Only 72 hours (3 days)
                        results.append({
                            'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                            'forecast_time': f"{forecast_time_hour:02d}:00",
                            'forecast_hour': hours_ahead,
                            'valid_time': valid_dt.strftime('%Y-%m-%d %H:%M:%S'),
                            'temperature': temp,
                            'source': 'open_meteo_historical'
                        })
            
            if results:
                logger.info(f"✓ Fetched {len(results)} hourly forecasts for {forecast_date} at {forecast_time_hour}:00")
                return results
            else:
                logger.warning(f"No forecast data available for {forecast_date}")
                return None
        else:
            logger.warning(f"No forecast data available for {forecast_date}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching forecast for {forecast_date}")
        return _retry_with_backoff(
            fetch_historical_forecast_for_date,
            forecast_date, forecast_time_hour, lat, lon, retry_count
        )
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API error fetching forecast for {forecast_date}: {e}")
        return _retry_with_backoff(
            fetch_historical_forecast_for_date,
            forecast_date, forecast_time_hour, lat, lon, retry_count
        )
        
    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"Error parsing response for {forecast_date}: {e}")
        return None


def _retry_with_backoff(func, forecast_date, forecast_time_hour, lat, lon, retry_count):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        forecast_date: Forecast date parameter
        forecast_time_hour: Forecast time parameter
        lat: Latitude parameter
        lon: Longitude parameter
        retry_count: Current retry attempt
    
    Returns:
        Result of function call or None if max retries exceeded
    """
    if retry_count >= MAX_RETRIES:
        logger.error(f"Max retries ({MAX_RETRIES}) exceeded for {forecast_date}")
        return None
    
    # Calculate exponential backoff delay
    delay = INITIAL_RETRY_DELAY * (2 ** retry_count)
    logger.info(f"Retrying in {delay} seconds... (attempt {retry_count + 1}/{MAX_RETRIES})")
    time_module.sleep(delay)
    
    return func(forecast_date, forecast_time_hour, lat, lon, retry_count + 1)


def fetch_historical_forecasts_batch(
    start_date,
    end_date=None,
    forecast_time_hours=None,
    lat=KLGA_LAT,
    lon=KLGA_LON,
    max_workers=10
):
    """
    Fetch historical forecasts for a range of dates using parallel requests.
    
    Args:
        start_date: First forecast issue date (datetime.date or string 'YYYY-MM-DD')
        end_date: Last forecast issue date (default: today)
        forecast_time_hours: List of hours when forecasts were issued (default: all 24 hours 0-23)
                            Can also be a single int for backward compatibility
        lat: Latitude (default: KLGA)
        lon: Longitude (default: KLGA)
        max_workers: Number of parallel threads (default: 10)
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - forecast_date
            - forecast_time
            - forecast_hour
            - valid_time
            - temperature
            - source
    """
    # Convert dates
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    
    if end_date is None:
        end_date = datetime.now().date()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Handle forecast_time_hours parameter
    if forecast_time_hours is None:
        forecast_time_hours = list(range(24))  # All 24 hours
    elif isinstance(forecast_time_hours, int):
        forecast_time_hours = [forecast_time_hours]  # Single hour
    
    logger.info(f"Fetching forecasts from {start_date} to {end_date}")
    logger.info(f"Forecast issue times: {len(forecast_time_hours)} hours per day ({forecast_time_hours[0]}:00 to {forecast_time_hours[-1]}:00)")
    logger.info(f"Using {max_workers} parallel workers")
    logger.info(f"Rate limit: {MAX_REQUESTS_PER_MINUTE} requests/minute")
    
    # Generate date and hour combinations
    date_hour_list = []
    current_date = start_date
    while current_date <= end_date:
        for hour in forecast_time_hours:
            date_hour_list.append((current_date, hour))
        current_date += timedelta(days=1)
    
    logger.info(f"Total forecast date-hour combinations to fetch: {len(date_hour_list)}")
    
    # Fetch forecasts in parallel
    all_forecasts = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_date_hour = {
            executor.submit(
                fetch_historical_forecast_for_date,
                date,
                hour,
                lat,
                lon
            ): (date, hour) for date, hour in date_hour_list
        }
        
        # Process completed tasks
        for future in as_completed(future_to_date_hour):
            date, hour = future_to_date_hour[future]
            try:
                result = future.result()
                if result:
                    all_forecasts.extend(result)  # result is a list of hourly forecasts
                completed += 1
                
                if completed % 50 == 0:
                    logger.info(f"Progress: {completed}/{len(date_hour_list)} date-hour combinations completed")
                    
            except Exception as e:
                logger.error(f"Error processing {date} {hour}:00: {e}")
    
    # Convert to DataFrame
    if all_forecasts:
        df = pd.DataFrame(all_forecasts)
        logger.info(f"✓ Successfully fetched {len(df)} hourly forecast records from {len(date_hour_list)} date-hour combinations")
        return df
    else:
        logger.warning("No forecasts retrieved")
        return pd.DataFrame(columns=[
            'forecast_date', 'forecast_time', 'forecast_hour',
            'valid_time', 'temperature', 'source'
        ])


def save_forecasts_to_csv(df, output_path='data/raw/historical_forecasts.csv'):
    """
    Save forecasts DataFrame to CSV file.
    
    Args:
        df: DataFrame with forecast data
        output_path: Path to save CSV file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved {len(df)} forecasts to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")
        return False


def load_existing_forecasts(csv_path='data/raw/historical_forecasts.csv'):
    """
    Load existing forecasts from CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        pd.DataFrame: Existing forecasts or empty DataFrame if file doesn't exist
    """
    try:
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} existing forecast records from {csv_path}")
            return df
        else:
            logger.info(f"No existing forecast file found at {csv_path}")
            return pd.DataFrame(columns=[
                'forecast_date', 'forecast_time', 'forecast_hour',
                'valid_time', 'temperature', 'source'
            ])
    except Exception as e:
        logger.error(f"Error loading existing forecasts: {e}")
        return pd.DataFrame(columns=[
            'forecast_date', 'forecast_time', 'forecast_hour',
            'valid_time', 'temperature', 'source'
        ])


def update_forecasts(
    start_date='2025-01-21',
    end_date=None,
    forecast_time_hours=None,
    csv_path='data/raw/historical_forecasts.csv'
):
    """
    Update forecast CSV with new data, avoiding duplicates.
    
    Args:
        start_date: First forecast issue date to fetch (default: 2025-01-21)
        end_date: Last forecast issue date to fetch (default: today)
        forecast_time_hours: List of hours to fetch (default: all 24 hours 0-23)
        csv_path: Path to CSV file
    
    Returns:
        pd.DataFrame: Updated forecasts DataFrame
    """
    # Load existing forecasts
    existing_df = load_existing_forecasts(csv_path)
    
    # Determine which date-hour combinations need to be fetched
    if len(existing_df) > 0:
        existing_df['forecast_date_parsed'] = pd.to_datetime(existing_df['forecast_date'])
        existing_combinations = set(
            zip(existing_df['forecast_date_parsed'].dt.date, 
                existing_df['forecast_time'].str.split(':').str[0].astype(int))
        )
        logger.info(f"Found {len(existing_combinations)} existing forecast date-hour combinations")
    else:
        existing_combinations = set()
    
    # Convert start/end dates
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if end_date is None:
        end_date = datetime.now().date()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Handle forecast_time_hours parameter
    if forecast_time_hours is None:
        forecast_time_hours = list(range(24))  # All 24 hours
    elif isinstance(forecast_time_hours, int):
        forecast_time_hours = [forecast_time_hours]
    
    # Filter to only date-hour combinations we don't have yet
    dates_to_fetch = []
    hours_to_fetch = []
    current = start_date
    while current <= end_date:
        for hour in forecast_time_hours:
            if (current, hour) not in existing_combinations:
                if not dates_to_fetch or dates_to_fetch[-1] != current:
                    dates_to_fetch.append(current)
                if hour not in hours_to_fetch:
                    hours_to_fetch.append(hour)
        current += timedelta(days=1)
    
    if not dates_to_fetch:
        logger.info("No new forecast date-hour combinations to fetch (all already exist)")
        return existing_df
    
    logger.info(f"Need to fetch forecasts for {len(dates_to_fetch)} dates with {len(forecast_time_hours)} hours each")
    
    # Fetch new forecasts for the filtered date range
    new_df = fetch_historical_forecasts_batch(
        dates_to_fetch[0],
        dates_to_fetch[-1] if len(dates_to_fetch) > 1 else dates_to_fetch[0],
        forecast_time_hours=forecast_time_hours
    )
    
    if len(new_df) > 0:
        logger.info(f"Adding {len(new_df)} new forecast records")
        
        # Combine with existing
        if len(existing_df) > 0:
            # Drop the temporary parsed column if it exists
            if 'forecast_date_parsed' in existing_df.columns:
                existing_df = existing_df.drop(columns=['forecast_date_parsed'])
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # Remove duplicates based on forecast_date, forecast_time, and valid_time
        combined_df = combined_df.drop_duplicates(
            subset=['forecast_date', 'forecast_time', 'valid_time'],
            keep='first'
        )
        
        # Sort by forecast date, forecast time, and valid time
        combined_df['forecast_date_parsed'] = pd.to_datetime(combined_df['forecast_date'])
        combined_df['valid_time_parsed'] = pd.to_datetime(combined_df['valid_time'])
        combined_df = combined_df.sort_values(['forecast_date_parsed', 'forecast_time', 'valid_time_parsed'])
        combined_df = combined_df.drop(columns=['forecast_date_parsed', 'valid_time_parsed'])
        
        # Save updated data
        save_forecasts_to_csv(combined_df, csv_path)
        
        return combined_df
    else:
        logger.warning("No new forecasts fetched")
        return existing_df


if __name__ == "__main__":
    print("=" * 70)
    print("Historical Forecast Fetcher")
    print("=" * 70)
    print("Fetching Open-Meteo historical forecasts for KLGA")
    print("Forecast issue times: Every hour (00:00 to 23:00 Eastern)")
    print("Forecast horizon: 72 hours (3 days) hourly")
    print()
    
    # Update forecasts from Jan 21, 2025 to present
    # Fetch all 24 hours per day (default behavior)
    df = update_forecasts(start_date='2025-01-21')
    
    if len(df) > 0:
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Total forecast records: {len(df)}")
        
        # Count unique forecast dates and times
        unique_dates = df['forecast_date'].nunique()
        unique_times = df['forecast_time'].nunique()
        print(f"Unique forecast issue dates: {unique_dates}")
        print(f"Unique forecast issue times per day: {unique_times}")
        print(f"Date range: {df['forecast_date'].min()} to {df['forecast_date'].max()}")
        
        # Show sample
        print(f"\nFirst few forecast records:")
        print(df.head(10))
        
        # Show statistics
        print(f"\nForecast issue times distribution:")
        print(df['forecast_time'].value_counts().sort_index())
        
        print(f"\nForecast hours distribution:")
        print(df['forecast_hour'].value_counts().sort_index().head(10))
        
        print(f"\nSaved to: data/raw/historical_forecasts.csv")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("No forecasts available")
        print("=" * 70)
