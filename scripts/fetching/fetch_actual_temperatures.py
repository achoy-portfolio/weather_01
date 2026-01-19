"""
Fetch actual hourly temperature observations from Open-Meteo Archive API.

This script retrieves historical weather observations at hourly intervals
to enable backtesting and ML model training.

API Documentation: https://open-meteo.com/en/docs/historical-weather-api
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
API_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1  # seconds
TIMEOUT = 30  # seconds
MAX_WORKERS = 10  # Concurrent requests
CHUNK_DAYS = 30  # Fetch 30 days per API call


def fetch_hourly_temperatures_chunk(
    start_date,
    end_date,
    lat=KLGA_LAT,
    lon=KLGA_LON,
    retry_count=0
):
    """
    Fetch hourly temperature observations for a date range.
    
    Args:
        start_date: First date to fetch (datetime.date or string 'YYYY-MM-DD')
        end_date: Last date to fetch (datetime.date or string 'YYYY-MM-DD')
        lat: Latitude (default: KLGA)
        lon: Longitude (default: KLGA)
        retry_count: Current retry attempt (used internally)
    
    Returns:
        pd.DataFrame: Hourly temperature data with columns:
            - timestamp: Datetime of observation
            - temperature_f: Temperature in Fahrenheit
        None: If fetch fails after all retries
    """
    # Convert dates to strings if needed
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
    
    start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date
    end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date
    
    # Build API parameters
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_str,
        'end_date': end_str,
        'hourly': 'temperature_2m',
        'temperature_unit': 'fahrenheit',
        'timezone': 'America/New_York'
    }
    
    try:
        logger.info(f"Fetching hourly temperatures from {start_str} to {end_str}")
        
        response = requests.get(API_BASE_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract hourly data
        if 'hourly' not in data or 'temperature_2m' not in data['hourly']:
            logger.warning(f"No hourly temperature data available for {start_str} to {end_str}")
            return None
        
        hourly_times = data['hourly']['time']
        hourly_temps = data['hourly']['temperature_2m']
        
        # Create DataFrame
        records = []
        for time_str, temp in zip(hourly_times, hourly_temps):
            if temp is not None:  # Skip missing data
                records.append({
                    'timestamp': time_str,
                    'temperature_f': temp
                })
        
        if not records:
            logger.warning(f"No valid temperature data for {start_str} to {end_str}")
            return None
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"✓ Retrieved {len(df)} hourly temperature readings")
        return df
            
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching temperatures for {start_str} to {end_str}")
        return _retry_with_backoff(
            fetch_hourly_temperatures_chunk,
            start_date, end_date, lat, lon, retry_count
        )
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API error fetching temperatures for {start_str} to {end_str}: {e}")
        return _retry_with_backoff(
            fetch_hourly_temperatures_chunk,
            start_date, end_date, lat, lon, retry_count
        )
        
    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"Error parsing response for {start_str} to {end_str}: {e}")
        return None


def _retry_with_backoff(func, start_date, end_date, lat, lon, retry_count):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        start_date: Start date parameter
        end_date: End date parameter
        lat: Latitude parameter
        lon: Longitude parameter
        retry_count: Current retry attempt
    
    Returns:
        Result of function call or None if max retries exceeded
    """
    if retry_count >= MAX_RETRIES:
        logger.error(f"Max retries ({MAX_RETRIES}) exceeded for {start_date} to {end_date}")
        return None
    
    # Calculate exponential backoff delay
    delay = INITIAL_RETRY_DELAY * (2 ** retry_count)
    logger.info(f"Retrying in {delay} seconds... (attempt {retry_count + 1}/{MAX_RETRIES})")
    time_module.sleep(delay)
    
    return func(start_date, end_date, lat, lon, retry_count + 1)


def fetch_hourly_temperatures_batch(
    start_date,
    end_date=None,
    lat=KLGA_LAT,
    lon=KLGA_LON
):
    """
    Fetch hourly temperatures for a range of dates using concurrent requests.
    
    Args:
        start_date: First date to fetch (datetime.date or string 'YYYY-MM-DD')
        end_date: Last date to fetch (default: yesterday, since today may be incomplete)
        lat: Latitude (default: KLGA)
        lon: Longitude (default: KLGA)
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - timestamp
            - temperature_f
    """
    # Convert dates
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    
    if end_date is None:
        # Default to yesterday (today's data may be incomplete)
        end_date = datetime.now().date() - timedelta(days=1)
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    logger.info(f"Fetching hourly temperatures from {start_date} to {end_date}")
    
    # Split into chunks for concurrent fetching
    chunks = []
    current_start = start_date
    
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS - 1), end_date)
        chunks.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)
    
    logger.info(f"Split into {len(chunks)} chunks of up to {CHUNK_DAYS} days each")
    
    # Fetch chunks concurrently
    all_data = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(
                fetch_hourly_temperatures_chunk,
                chunk_start,
                chunk_end,
                lat,
                lon
            ): (chunk_start, chunk_end)
            for chunk_start, chunk_end in chunks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            chunk_start, chunk_end = future_to_chunk[future]
            try:
                df_chunk = future.result()
                if df_chunk is not None and len(df_chunk) > 0:
                    all_data.append(df_chunk)
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_start} to {chunk_end}: {e}")
    
    # Combine all chunks
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"✓ Successfully fetched {len(df)} hourly temperature records")
        return df
    else:
        logger.warning("No temperature data retrieved")
        return pd.DataFrame(columns=['timestamp', 'temperature_f'])


def save_temperatures_to_csv(df, output_path='data/raw/actual_temperatures.csv'):
    """
    Save temperatures DataFrame to CSV file.
    
    Args:
        df: DataFrame with temperature data
        output_path: Path to save CSV file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved {len(df)} hourly temperature records to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")
        return False


def load_existing_temperatures(csv_path='data/raw/actual_temperatures.csv'):
    """
    Load existing temperature data from CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        pd.DataFrame: Existing temperatures or empty DataFrame if file doesn't exist
    """
    try:
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"Loaded {len(df)} existing hourly temperature records from {csv_path}")
            return df
        else:
            logger.info(f"No existing temperature file found at {csv_path}")
            return pd.DataFrame(columns=['timestamp', 'temperature_f'])
    except Exception as e:
        logger.error(f"Error loading existing temperatures: {e}")
        return pd.DataFrame(columns=['timestamp', 'temperature_f'])


def update_temperatures(
    start_date='2025-01-21',
    end_date=None,
    csv_path='data/raw/actual_temperatures.csv'
):
    """
    Update temperature CSV with new data, avoiding duplicates.
    
    Args:
        start_date: First date to fetch (default: 2025-01-21)
        end_date: Last date to fetch (default: yesterday)
        csv_path: Path to CSV file
    
    Returns:
        pd.DataFrame: Updated temperatures DataFrame
    """
    # Load existing temperatures
    existing_df = load_existing_temperatures(csv_path)
    
    # Determine which dates need to be fetched
    if len(existing_df) > 0:
        existing_timestamps = set(existing_df['timestamp'])
        logger.info(f"Found {len(existing_timestamps)} existing hourly temperature records")
        
        # Find the latest timestamp to determine where to start fetching
        latest_timestamp = existing_df['timestamp'].max()
        fetch_start = (latest_timestamp + timedelta(hours=1)).date()
        
        # Convert start_date to date object
        if isinstance(start_date, str):
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        else:
            start_date_obj = start_date
        
        # Use the later of the two dates
        fetch_start = max(fetch_start, start_date_obj)
        
        logger.info(f"Latest existing data: {latest_timestamp}, fetching from {fetch_start}")
    else:
        fetch_start = start_date
        existing_timestamps = set()
    
    # Fetch new temperatures
    new_df = fetch_hourly_temperatures_batch(fetch_start, end_date)
    
    if len(new_df) > 0:
        # Filter out timestamps that already exist
        new_df = new_df[~new_df['timestamp'].isin(existing_timestamps)]
        
        if len(new_df) > 0:
            logger.info(f"Adding {len(new_df)} new hourly temperature records")
            
            # Combine with existing
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Sort by timestamp
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # Save updated data
            save_temperatures_to_csv(combined_df, csv_path)
            
            return combined_df
        else:
            logger.info("No new temperature records to add (all timestamps already exist)")
            return existing_df
    else:
        logger.warning("No new temperature data fetched")
        return existing_df


if __name__ == "__main__":
    print("=" * 70)
    print("Actual Hourly Temperature Fetcher")
    print("=" * 70)
    print("Fetching Open-Meteo Archive API hourly temperature observations for KLGA")
    print()
    
    # Update temperatures from Jan 21, 2025 to present
    df = update_temperatures(start_date='2025-01-21')
    
    if len(df) > 0:
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Total hourly temperature records: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\nFirst few records:")
        print(df.head(10))
        print(f"\nLast few records:")
        print(df.tail(10))
        print(f"\nTemperature statistics:")
        print(df['temperature_f'].describe())
        print(f"\nSaved to: data/raw/actual_temperatures.csv")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("No temperature data available")
        print("=" * 70)
