"""
Fetch actual hourly temperatures from Weather Underground using Playwright.

This script scrapes historical weather data from Weather Underground,
which is the source Polymarket uses to resolve temperature bets.

Outputs:
1. Hourly temperature data (same format as Open-Meteo)
2. Daily maximum temperatures

Example URL: https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA/date/2025-4-25
"""

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time as time_module
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA/date"
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2  # seconds
PAGE_LOAD_TIMEOUT = 30000  # milliseconds
MAX_WORKERS = 15  # Concurrent browser contexts (be respectful)
REQUEST_DELAY = 0.2  # Delay between requests in seconds


def fetch_hourly_temps(date, retry_count=0):
    """
    Fetch hourly temperatures from Weather Underground for a specific date.
    
    Args:
        date: Date to fetch (datetime.date or string 'YYYY-MM-DD')
        retry_count: Current retry attempt (used internally)
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - timestamp: Datetime of observation (YYYY-MM-DD HH:MM)
            - temperature_f: Temperature in Fahrenheit
        None: If fetch fails after all retries
    """
    # Convert date to proper format
    if isinstance(date, str):
        date_obj = datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, datetime):
        date_obj = date.date()
    else:
        date_obj = date
    
    # Format URL: YYYY-M-D (no leading zeros)
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    url = f"{BASE_URL}/{year}-{month}-{day}"
    
    try:
        logger.info(f"Fetching hourly temperatures for {date_obj} from Weather Underground")
        
        # Add delay to be respectful
        time_module.sleep(REQUEST_DELAY)
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Navigate to the page
            try:
                page.goto(url, timeout=PAGE_LOAD_TIMEOUT, wait_until='domcontentloaded')
                
                # Wait a bit for dynamic content (reduced from 3s to 2s)
                page.wait_for_timeout(2000)
                
                # Wait for the observations table to load
                page.wait_for_selector('table', timeout=8000)
            except PlaywrightTimeout:
                logger.warning(f"Timeout waiting for page/table to load for {date_obj}")
                browser.close()
                return None
            except Exception as e:
                logger.error(f"Error loading page for {date_obj}: {e}")
                browser.close()
                return None
            
            # Extract hourly data from the observations table
            hourly_data = []
            
            # Get all tables on the page
            tables = page.query_selector_all('table')
            
            for table in tables:
                # Get all rows in the table
                rows = table.query_selector_all('tr')
                
                for row in rows:
                    cells = row.query_selector_all('td, th')
                    
                    if len(cells) >= 2:
                        # First cell often contains time
                        time_text = cells[0].inner_text().strip()
                        
                        # Check if this looks like a time (e.g., "12:00 AM", "1:00 PM")
                        time_match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)', time_text, re.IGNORECASE)
                        
                        if time_match:
                            hour = int(time_match.group(1))
                            minute = int(time_match.group(2))
                            am_pm = time_match.group(3).upper()
                            
                            # Convert to 24-hour format
                            if am_pm == 'PM' and hour != 12:
                                hour += 12
                            elif am_pm == 'AM' and hour == 12:
                                hour = 0
                            
                            # Look for temperature in subsequent cells
                            for cell in cells[1:]:
                                temp_text = cell.inner_text().strip()
                                # Extract temperature (look for number, possibly with °)
                                temp_match = re.search(r'(-?\d+\.?\d*)\s*°?', temp_text)
                                if temp_match:
                                    try:
                                        temp = float(temp_match.group(1))
                                        
                                        # Determine the actual date for this reading
                                        # If time is late PM (10 PM - 11:59 PM), it's from the previous day
                                        # If time is early AM (12 AM - 5:59 AM), it could be from the next day
                                        actual_date = date_obj
                                        
                                        if am_pm == 'PM' and hour >= 22:  # 10 PM or later
                                            # This reading is from the previous day
                                            actual_date = date_obj - timedelta(days=1)
                                        elif am_pm == 'AM' and hour < 6:  # Before 6 AM
                                            # This reading is from the next day
                                            actual_date = date_obj + timedelta(days=1)
                                        
                                        # Create timestamp with adjusted date
                                        timestamp = datetime(actual_date.year, actual_date.month, actual_date.day, hour, minute)
                                        
                                        hourly_data.append({
                                            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M'),
                                            'temperature_f': temp
                                        })
                                        break
                                    except ValueError:
                                        pass
            
            browser.close()
            
            if hourly_data:
                df = pd.DataFrame(hourly_data)
                df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
                logger.info(f"✓ Retrieved {len(df)} hourly temperature readings for {date_obj}")
                return df
            else:
                logger.warning(f"Could not find hourly temperature data on page for {date_obj}")
                return None
                
    except Exception as e:
        logger.error(f"Error fetching temperature for {date_obj}: {e}")
        return _retry_with_backoff(fetch_hourly_temps, date, retry_count)


def fetch_daily_max_temp(date, retry_count=0):
    """
    Fetch the daily maximum temperature directly from Weather Underground page.
    This is the official max that Polymarket uses for resolution.
    
    Args:
        date: Date to fetch (datetime.date or string 'YYYY-MM-DD')
        retry_count: Current retry attempt (used internally)
    
    Returns:
        dict: Dictionary with keys:
            - date: Date string (YYYY-MM-DD)
            - max_temp_f: Maximum temperature in Fahrenheit
        None: If fetch fails after all retries
    """
    # Convert date to proper format
    if isinstance(date, str):
        date_obj = datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, datetime):
        date_obj = date.date()
    else:
        date_obj = date
    
    # Format URL: YYYY-M-D (no leading zeros)
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    url = f"{BASE_URL}/{year}-{month}-{day}"
    
    try:
        logger.info(f"Fetching daily max temperature for {date_obj} from Weather Underground")
        
        # Add delay to be respectful
        time_module.sleep(REQUEST_DELAY)
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Navigate to the page
            try:
                page.goto(url, timeout=PAGE_LOAD_TIMEOUT, wait_until='domcontentloaded')
                
                # Wait a bit for dynamic content (reduced from 3s to 2s)
                page.wait_for_timeout(2000)
                
                # Wait for content to load
                page.wait_for_selector('body', timeout=8000)
            except PlaywrightTimeout:
                logger.warning(f"Timeout waiting for page to load for {date_obj}")
                browser.close()
                return None
            except Exception as e:
                logger.error(f"Error loading page for {date_obj}: {e}")
                browser.close()
                return None
            
            # Get the full page content
            content = page.content()
            
            # Find the daily maximum temperature
            max_temp = None
            
            # Method 1: Look for "High Temp" or similar in tables
            tables = page.query_selector_all('table')
            for table in tables:
                rows = table.query_selector_all('tr')
                for row in rows:
                    cells = row.query_selector_all('td, th')
                    for i, cell in enumerate(cells):
                        cell_text = cell.inner_text().strip()
                        if re.search(r'(?:High|Max|Maximum)\s*(?:Temp|Temperature)?', cell_text, re.IGNORECASE):
                            # Look for temperature in next cell or same cell
                            if i + 1 < len(cells):
                                temp_text = cells[i + 1].inner_text().strip()
                            else:
                                temp_text = cell_text
                            
                            temp_match = re.search(r'(-?\d+\.?\d*)\s*°?F?', temp_text)
                            if temp_match:
                                try:
                                    max_temp = float(temp_match.group(1))
                                    break
                                except ValueError:
                                    pass
                    if max_temp is not None:
                        break
                if max_temp is not None:
                    break
            
            # Method 2: Look in divs for summary information
            if max_temp is None:
                divs = page.query_selector_all('div')
                for div in divs:
                    text = div.inner_text().strip()
                    # Look for patterns like "High: 45°" or "Max: 45"
                    high_match = re.search(r'(?:High|Max|Maximum)[\s:]*(-?\d+\.?\d*)\s*°?F?', text, re.IGNORECASE)
                    if high_match:
                        try:
                            max_temp = float(high_match.group(1))
                            break
                        except ValueError:
                            pass
            
            browser.close()
            
            if max_temp is not None:
                logger.info(f"✓ Retrieved daily max temperature: {max_temp}°F for {date_obj}")
                return {
                    'date': date_obj.strftime('%Y-%m-%d'),
                    'max_temp_f': max_temp
                }
            else:
                logger.warning(f"Could not find daily max temperature on page for {date_obj}")
                return None
                
    except Exception as e:
        logger.error(f"Error fetching temperature for {date_obj}: {e}")
        return _retry_with_backoff(fetch_daily_max_temp, date, retry_count)


def _retry_with_backoff(func, date, retry_count):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        date: Date parameter
        retry_count: Current retry attempt
    
    Returns:
        Result of function call or None if max retries exceeded
    """
    if retry_count >= MAX_RETRIES:
        logger.error(f"Max retries ({MAX_RETRIES}) exceeded for {date}")
        return None
    
    # Calculate exponential backoff delay
    delay = INITIAL_RETRY_DELAY * (2 ** retry_count)
    logger.info(f"Retrying in {delay} seconds... (attempt {retry_count + 1}/{MAX_RETRIES})")
    time_module.sleep(delay)
    
    return func(date, retry_count + 1)


def fetch_hourly_temps_batch(start_date, end_date=None):
    """
    Fetch hourly temperatures for a range of dates.
    
    Args:
        start_date: First date to fetch (datetime.date or string 'YYYY-MM-DD')
        end_date: Last date to fetch (default: yesterday)
    
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
    
    # Generate list of dates
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    
    logger.info(f"Fetching {len(dates)} days of data")
    
    # Fetch data with limited concurrency
    all_data = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_date = {
            executor.submit(fetch_hourly_temps, date): date
            for date in dates
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            completed += 1
            try:
                result = future.result()
                if result is not None and len(result) > 0:
                    all_data.append(result)
                logger.info(f"Progress: {completed}/{len(dates)} days completed")
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}")
    
    # Combine all data
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"✓ Successfully fetched {len(df)} hourly temperature records")
        return df
    else:
        logger.warning("No temperature data retrieved")
        return pd.DataFrame(columns=['timestamp', 'temperature_f'])


def fetch_daily_max_temps_batch(start_date, end_date=None):
    """
    Fetch daily maximum temperatures for a range of dates.
    
    Args:
        start_date: First date to fetch (datetime.date or string 'YYYY-MM-DD')
        end_date: Last date to fetch (default: yesterday)
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - date
            - max_temp_f
    """
    # Convert dates
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    
    if end_date is None:
        # Default to yesterday (today's data may be incomplete)
        end_date = datetime.now().date() - timedelta(days=1)
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    logger.info(f"Fetching daily max temperatures from {start_date} to {end_date}")
    
    # Generate list of dates
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    
    logger.info(f"Fetching {len(dates)} days of data")
    
    # Fetch data with limited concurrency
    all_data = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_date = {
            executor.submit(fetch_daily_max_temp, date): date
            for date in dates
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            completed += 1
            try:
                result = future.result()
                if result is not None:
                    all_data.append(result)
                logger.info(f"Progress: {completed}/{len(dates)} days completed")
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}")
    
    # Create DataFrame
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sort_values('date').reset_index(drop=True)
        logger.info(f"✓ Successfully fetched {len(df)} daily max temperature records")
        return df
    else:
        logger.warning("No temperature data retrieved")
        return pd.DataFrame(columns=['date', 'max_temp_f'])


def save_temperatures_to_csv(df, output_path):
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
        logger.info(f"✓ Saved {len(df)} records to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")
        return False


def load_existing_hourly_temps(csv_path='data/raw/wunderground_hourly_temps.csv'):
    """
    Load existing hourly temperature data from CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        pd.DataFrame: Existing temperatures or empty DataFrame if file doesn't exist
    """
    try:
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} existing hourly temperature records from {csv_path}")
            return df
        else:
            logger.info(f"No existing temperature file found at {csv_path}")
            return pd.DataFrame(columns=['timestamp', 'temperature_f'])
    except Exception as e:
        logger.error(f"Error loading existing temperatures: {e}")
        return pd.DataFrame(columns=['timestamp', 'temperature_f'])


def load_existing_daily_max(csv_path='data/raw/wunderground_daily_max_temps.csv'):
    """
    Load existing daily max temperature data from CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        pd.DataFrame: Existing temperatures or empty DataFrame if file doesn't exist
    """
    try:
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} existing daily max temperature records from {csv_path}")
            return df
        else:
            logger.info(f"No existing temperature file found at {csv_path}")
            return pd.DataFrame(columns=['date', 'max_temp_f'])
    except Exception as e:
        logger.error(f"Error loading existing temperatures: {e}")
        return pd.DataFrame(columns=['date', 'max_temp_f'])


def update_temperatures(
    start_date='2025-01-21',
    end_date=None,
    hourly_csv_path='data/raw/wunderground_hourly_temps.csv',
    daily_csv_path='data/raw/wunderground_daily_max_temps.csv'
):
    """
    Update temperature CSVs with new data, avoiding duplicates.
    Fetches both hourly data and daily max directly from Weather Underground.
    
    Args:
        start_date: First date to fetch (default: 2025-01-21)
        end_date: Last date to fetch (default: yesterday)
        hourly_csv_path: Path to hourly CSV file
        daily_csv_path: Path to daily max CSV file
    
    Returns:
        tuple: (hourly_df, daily_max_df)
    """
    # Load existing data
    existing_hourly = load_existing_hourly_temps(hourly_csv_path)
    existing_daily = load_existing_daily_max(daily_csv_path)
    
    # Determine which dates need to be fetched for hourly data
    if len(existing_hourly) > 0:
        existing_timestamps = set(existing_hourly['timestamp'])
        logger.info(f"Found {len(existing_timestamps)} existing hourly temperature records")
        
        # Find the latest timestamp to determine where to start fetching
        latest_timestamp = pd.to_datetime(existing_hourly['timestamp']).max()
        fetch_start_hourly = (latest_timestamp + timedelta(hours=1)).date()
        
        # Convert start_date to date object
        if isinstance(start_date, str):
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        else:
            start_date_obj = start_date
        
        # Use the later of the two dates
        fetch_start_hourly = max(fetch_start_hourly, start_date_obj)
        
        logger.info(f"Latest existing hourly data: {latest_timestamp}, fetching from {fetch_start_hourly}")
    else:
        fetch_start_hourly = start_date
        existing_timestamps = set()
    
    # Determine which dates need to be fetched for daily max
    if len(existing_daily) > 0:
        existing_dates = set(existing_daily['date'])
        logger.info(f"Found {len(existing_dates)} existing daily max temperature records")
        
        # Find the latest date
        latest_date_str = existing_daily['date'].max()
        latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d').date()
        fetch_start_daily = latest_date + timedelta(days=1)
        
        # Convert start_date to date object
        if isinstance(start_date, str):
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        else:
            start_date_obj = start_date
        
        # Use the later of the two dates
        fetch_start_daily = max(fetch_start_daily, start_date_obj)
        
        logger.info(f"Latest existing daily data: {latest_date}, fetching from {fetch_start_daily}")
    else:
        fetch_start_daily = start_date
        existing_dates = set()
    
    # Fetch new hourly temperatures
    new_hourly = fetch_hourly_temps_batch(fetch_start_hourly, end_date)
    
    if len(new_hourly) > 0:
        # Filter out timestamps that already exist
        new_hourly = new_hourly[~new_hourly['timestamp'].isin(existing_timestamps)]
        
        if len(new_hourly) > 0:
            logger.info(f"Adding {len(new_hourly)} new hourly temperature records")
            
            # Combine with existing
            combined_hourly = pd.concat([existing_hourly, new_hourly], ignore_index=True)
            
            # Sort by timestamp
            combined_hourly = combined_hourly.sort_values('timestamp').reset_index(drop=True)
            
            # Save updated hourly data
            save_temperatures_to_csv(combined_hourly, hourly_csv_path)
        else:
            logger.info("No new hourly temperature records to add (all timestamps already exist)")
            combined_hourly = existing_hourly
    else:
        logger.warning("No new hourly temperature data fetched")
        combined_hourly = existing_hourly
    
    # Fetch new daily max temperatures (directly from website, not calculated)
    new_daily = fetch_daily_max_temps_batch(fetch_start_daily, end_date)
    
    if len(new_daily) > 0:
        # Filter out dates that already exist
        new_daily = new_daily[~new_daily['date'].isin(existing_dates)]
        
        if len(new_daily) > 0:
            logger.info(f"Adding {len(new_daily)} new daily max temperature records")
            
            # Combine with existing
            combined_daily = pd.concat([existing_daily, new_daily], ignore_index=True)
            
            # Sort by date
            combined_daily = combined_daily.sort_values('date').reset_index(drop=True)
            
            # Save updated daily data
            save_temperatures_to_csv(combined_daily, daily_csv_path)
        else:
            logger.info("No new daily max temperature records to add (all dates already exist)")
            combined_daily = existing_daily
    else:
        logger.warning("No new daily max temperature data fetched")
        combined_daily = existing_daily
    
    return combined_hourly, combined_daily


if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Weather Underground Temperature Fetcher (Playwright)")
    print("=" * 70)
    print("Scraping hourly temperatures from Weather Underground for KLGA")
    print("(This is the source Polymarket uses for bet resolution)")
    print()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        start_date = sys.argv[1]
        end_date = sys.argv[2] if len(sys.argv) > 2 else None
        print(f"Using start date from command line: {start_date}")
        if end_date:
            print(f"Using end date from command line: {end_date}")
    else:
        # Default: fetch last 30 days only
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"No dates provided, fetching last 30 days: {start_date} to {end_date}")
        print("To fetch specific dates, run:")
        print("  python fetch_wunderground_actual.py YYYY-MM-DD [YYYY-MM-DD]")
        print()
    
    # Update temperatures
    hourly_df, daily_max_df = update_temperatures(start_date=start_date, end_date=end_date)
    
    if len(hourly_df) > 0:
        print("\n" + "=" * 70)
        print("Hourly Temperature Summary")
        print("=" * 70)
        print(f"Total hourly temperature records: {len(hourly_df)}")
        print(f"Timestamp range: {hourly_df['timestamp'].min()} to {hourly_df['timestamp'].max()}")
        print(f"\nFirst few records:")
        print(hourly_df.head(10))
        print(f"\nLast few records:")
        print(hourly_df.tail(10))
        print(f"\nTemperature statistics:")
        print(hourly_df['temperature_f'].describe())
        print(f"\nSaved to: data/raw/wunderground_hourly_temps.csv")
    
    if len(daily_max_df) > 0:
        print("\n" + "=" * 70)
        print("Daily Maximum Temperature Summary")
        print("=" * 70)
        print(f"Total daily records: {len(daily_max_df)}")
        print(f"Date range: {daily_max_df['date'].min()} to {daily_max_df['date'].max()}")
        print(f"\nFirst few records:")
        print(daily_max_df.head(10))
        print(f"\nLast few records:")
        print(daily_max_df.tail(10))
        print(f"\nTemperature statistics:")
        print(daily_max_df['max_temp_f'].describe())
        print(f"\nSaved to: data/raw/wunderground_daily_max_temps.csv")
    
    if len(hourly_df) == 0 and len(daily_max_df) == 0:
        print("\n" + "=" * 70)
        print("No temperature data available")
        print("=" * 70)
    
    print("=" * 70)
