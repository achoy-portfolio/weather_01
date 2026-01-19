"""
Enhanced Polymarket Historical Odds Fetcher for Backtest System.

Fetches historical odds for all NYC temperature markets in a date range.
Designed for backtesting betting strategies.

Requirements covered: 3.1, 3.2, 3.3, 3.4, 3.5

Run: python scripts/fetching/fetch_polymarket_historical_enhanced.py
"""

import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import sys
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

sys.path.insert(0, '.')

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")

# Configuration
MAX_WORKERS = 5  # Concurrent requests (be respectful to Polymarket API)
REQUEST_TIMEOUT = 10  # seconds


def generate_event_slug(target_date):
    """
    Generate Polymarket event slug for a specific date.
    
    Args:
        target_date: datetime object for the event date
    
    Returns:
        Event slug string (e.g., 'highest-temperature-in-nyc-on-january-22')
    """
    month = target_date.strftime('%B').lower()
    day = target_date.day
    
    slug = f"highest-temperature-in-nyc-on-{month}-{day}"
    
    return slug


def generate_event_slugs_for_range(start_date, end_date):
    """
    Generate event slugs for a date range.
    
    Args:
        start_date: datetime object for first date
        end_date: datetime object for last date
    
    Returns:
        List of tuples (date, slug)
    """
    slugs = []
    current_date = start_date
    
    while current_date <= end_date:
        slug = generate_event_slug(current_date)
        slugs.append((current_date, slug))
        current_date += timedelta(days=1)
    
    return slugs


def parse_threshold_from_question(question):
    """
    Parse temperature threshold from market question.
    Handles "≥", "≤", and range formats.
    
    Args:
        question: Market question string
    
    Returns:
        Tuple of (threshold, threshold_type, threshold_display)
        threshold_type: 'above', 'below', or 'range'
    """
    # Check for range pattern: "between 34-35°F" or "34-35°F"
    range_match = re.search(r'(\d+)-(\d+)°F', question)
    if range_match:
        low = int(range_match.group(1))
        high = int(range_match.group(2))
        threshold = f"{low}-{high}"
        threshold_type = "range"
        threshold_display = f"{low}-{high}°F"
        return threshold, threshold_type, threshold_display
    
    # Check for "or below" / "or lower" pattern: "33°F or below"
    if 'or below' in question.lower() or 'or lower' in question.lower():
        temp_match = re.search(r'(\d+)°F', question)
        if temp_match:
            temp = int(temp_match.group(1))
            threshold = f"≤{temp}"
            threshold_type = "below"
            threshold_display = f"≤{temp}°F"
            return threshold, threshold_type, threshold_display
    
    # Check for "or higher" / "or above" pattern: "44°F or higher"
    if 'or higher' in question.lower() or 'or above' in question.lower():
        temp_match = re.search(r'(\d+)°F', question)
        if temp_match:
            temp = int(temp_match.group(1))
            threshold = f"≥{temp}"
            threshold_type = "above"
            threshold_display = f"≥{temp}°F"
            return threshold, threshold_type, threshold_display
    
    # Fallback: just extract first number (shouldn't happen with proper markets)
    temp_match = re.search(r'(\d+)°F', question)
    if temp_match:
        temp = int(temp_match.group(1))
        threshold = str(temp)
        threshold_type = "unknown"
        threshold_display = f"{temp}°F"
        return threshold, threshold_type, threshold_display
    
    return None, None, "Unknown"


def fetch_market_info(event_slug):
    """
    Fetch market data to get token IDs and threshold information.
    
    Args:
        event_slug: The event identifier from URL
    
    Returns:
        List of dicts with market info and token IDs, or None if error
    """
    api_url = f"https://gamma-api.polymarket.com/events?slug={event_slug}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(api_url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or len(data) == 0:
            return None
        
        event = data[0]
        markets = event.get('markets', [])
        
        if not markets:
            return None
        
        # Extract token IDs and market info
        market_info = []
        for market in markets:
            question = market.get('question', '')
            
            # Parse threshold from question
            threshold, threshold_type, threshold_display = parse_threshold_from_question(question)
            
            if threshold is None:
                continue
            
            # Get token IDs (Polymarket has YES and NO tokens)
            clobTokenIds = market.get('clobTokenIds')
            
            # Parse token IDs
            token_ids = []
            if clobTokenIds:
                try:
                    if isinstance(clobTokenIds, str):
                        token_ids = json.loads(clobTokenIds)
                    else:
                        token_ids = clobTokenIds
                except:
                    pass
            
            if token_ids and len(token_ids) >= 1:
                # Get volume from market
                volume = float(market.get('volume', 0)) if market.get('volume') else 0
                
                market_info.append({
                    'question': question,
                    'threshold': threshold,
                    'threshold_type': threshold_type,
                    'threshold_display': threshold_display,
                    'market_id': market.get('id'),
                    'condition_id': market.get('conditionId'),
                    'yes_token_id': token_ids[0],  # First token is YES
                    'no_token_id': token_ids[1] if len(token_ids) > 1 else None,
                    'volume': volume
                })
        
        return market_info
        
    except Exception as e:
        return None


def fetch_all_odds_history(token_id, target_date, volume=0):
    """
    Fetch ALL historical odds data points for a market (not just market open).
    
    Args:
        token_id: The CLOB token ID
        target_date: datetime object for the event date
        volume: Market volume (for record keeping)
    
    Returns:
        List of dicts with all odds data points, or empty list if error
    """
    # Markets typically open 2 days before, fetch from 3 days before to event day
    start_ts = (target_date - timedelta(days=3)).timestamp()
    end_ts = target_date.timestamp()
    
    url = "https://clob.polymarket.com/prices-history"
    
    params = {
        'market': token_id,
        'startTs': int(start_ts),
        'endTs': int(end_ts),
        'fidelity': 5  # 5-minute resolution for all available data points
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        history = data.get('history', [])
        
        if not history:
            return []
        
        # Return ALL data points, not just the first one
        all_points = []
        for point in history:
            fetch_timestamp = datetime.fromtimestamp(point['t'], tz=NY_TZ)
            yes_probability = point['p']
            
            all_points.append({
                'fetch_timestamp': fetch_timestamp,
                'yes_probability': yes_probability,
                'volume': volume
            })
        
        return all_points
        
    except Exception as e:
        return []


def fetch_all_odds_for_date(target_date):
    """
    Fetch all market odds for a specific date (all historical data points).
    
    Args:
        target_date: datetime object for the event date
    
    Returns:
        List of dicts with odds data for all thresholds and all timestamps
    """
    slug = generate_event_slug(target_date)
    
    logger.info(f"Fetching {target_date.strftime('%Y-%m-%d')}: {slug}")
    
    # Get market info
    markets = fetch_market_info(slug)
    
    if not markets:
        logger.warning(f"  ✗ No markets found for {target_date.strftime('%Y-%m-%d')}")
        return []
    
    logger.info(f"  ✓ Found {len(markets)} markets")
    
    # Fetch odds for each market
    odds_records = []
    
    for market in markets:
        threshold = market['threshold']
        threshold_type = market['threshold_type']
        token_id = market['yes_token_id']
        volume = market['volume']
        
        # Fetch ALL historical odds data points
        odds_data_points = fetch_all_odds_history(token_id, target_date, volume)
        
        if odds_data_points:
            # Create a record for each data point
            for odds_data in odds_data_points:
                odds_records.append({
                    'event_date': target_date.strftime('%Y-%m-%d'),
                    'fetch_timestamp': odds_data['fetch_timestamp'].isoformat(),
                    'threshold': threshold,
                    'threshold_type': threshold_type,
                    'yes_probability': odds_data['yes_probability'],
                    'volume': odds_data['volume']
                })
        
        # Small delay to be respectful to API
        time.sleep(0.05)
    
    if odds_records:
        logger.info(f"  ✓ Retrieved {len(odds_records)} total odds data points")
    
    return odds_records


def fetch_odds_for_date_range(start_date, end_date, output_file='data/raw/polymarket_odds_history.csv'):
    """
    Fetch historical odds for all dates in a range using concurrent requests.
    
    Args:
        start_date: datetime object or string (YYYY-MM-DD)
        end_date: datetime object or string (YYYY-MM-DD)
        output_file: Path to save CSV file
    
    Returns:
        DataFrame with all odds data
    """
    # Parse dates if strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    logger.info(f"\nFetching odds for date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Total days: {(end_date - start_date).days + 1}\n")
    
    # Generate list of dates
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Fetch odds for all dates concurrently
    all_odds = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_date = {
            executor.submit(fetch_all_odds_for_date, date): date
            for date in dates
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            try:
                odds_records = future.result()
                if odds_records:
                    all_odds.extend(odds_records)
            except Exception as e:
                logger.error(f"Error processing {date.strftime('%Y-%m-%d')}: {e}")
    
    if not all_odds:
        logger.warning("\n✗ No odds data retrieved")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_odds)
    
    # Sort by event date and threshold
    df = df.sort_values(['event_date', 'threshold'])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✓ Historical odds saved to: {output_file}")
    logger.info(f"  Total records: {len(df)}")
    logger.info(f"  Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    logger.info(f"  Unique dates: {df['event_date'].nunique()}")
    logger.info(f"{'='*70}")
    
    return df


def display_odds_summary(df):
    """Display summary of fetched odds."""
    if df.empty:
        return
    
    print("\nOdds Summary:")
    print("-" * 70)
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    print(f"Unique dates: {df['event_date'].nunique()}")
    print(f"\nThreshold types:")
    print(df['threshold_type'].value_counts())
    print(f"\nSample data (first 5 rows):")
    print(df.head())
    print("-" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced Polymarket Historical Odds Fetcher")
    print("=" * 70)
    print()
    
    # Parse command line arguments
    if len(sys.argv) >= 3:
        # python script.py 2025-01-22 2025-02-15
        start_date_str = sys.argv[1]
        end_date_str = sys.argv[2]
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    else:
        # Default: Jan 22, 2025 to Jan 10, 2026 (as per requirements)
        start_date = datetime(2025, 1, 22)
        end_date = datetime(2026, 1, 10)
        print("Using default date range (Jan 22, 2025 - Jan 10, 2026)")
        print("To specify custom range: python script.py YYYY-MM-DD YYYY-MM-DD\n")
    
    # Fetch odds for date range
    df = fetch_odds_for_date_range(start_date, end_date)
    
    if not df.empty:
        display_odds_summary(df)
        print("\n✓ Success! Historical odds retrieved and saved.")
    else:
        print("\n✗ No data retrieved. Markets may not exist for this date range.")
