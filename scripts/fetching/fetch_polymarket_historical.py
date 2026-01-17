"""
Fetch historical odds from Polymarket for temperature markets.
Tracks how odds changed over time leading up to the event.

Run: python scripts/fetching/fetch_polymarket_historical.py
"""

import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import sys

sys.path.insert(0, '.')

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")


def fetch_market_token_ids(event_slug):
    """
    Fetch market data to get token IDs for historical price queries.
    
    Args:
        event_slug: The event identifier from URL
    
    Returns:
        List of dicts with market info and token IDs
    """
    api_url = f"https://gamma-api.polymarket.com/events?slug={event_slug}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    print(f"Fetching market info for: {event_slug}")
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or len(data) == 0:
            print("✗ Event not found")
            return None
        
        event = data[0]
        markets = event.get('markets', [])
        
        if not markets:
            print("✗ No markets found")
            return None
        
        print(f"✓ Found {len(markets)} markets")
        
        # Extract token IDs and market info
        market_info = []
        for market in markets:
            question = market.get('question', '')
            
            # Extract temperature range from question
            import re
            
            # Check for range pattern: "between 34-35°F"
            range_match = re.search(r'between (\d+)-(\d+)°F', question)
            if range_match:
                low = int(range_match.group(1))
                high = int(range_match.group(2))
                threshold = f"{low}-{high}"
                threshold_display = f"{low}-{high}°F"
            # Check for "or below" pattern: "33°F or below"
            elif 'or below' in question.lower() or 'or lower' in question.lower():
                temp_match = re.search(r'(\d+)°F', question)
                if temp_match:
                    temp = int(temp_match.group(1))
                    threshold = f"≤{temp}"
                    threshold_display = f"≤{temp}°F"
                else:
                    threshold = None
                    threshold_display = "Unknown"
            # Check for "or higher" pattern: "44°F or higher"
            elif 'or higher' in question.lower() or 'or above' in question.lower():
                temp_match = re.search(r'(\d+)°F', question)
                if temp_match:
                    temp = int(temp_match.group(1))
                    threshold = f"≥{temp}"
                    threshold_display = f"≥{temp}°F"
                else:
                    threshold = None
                    threshold_display = "Unknown"
            else:
                # Fallback: just extract first number
                temp_match = re.search(r'(\d+)°F', question)
                threshold = int(temp_match.group(1)) if temp_match else None
                threshold_display = f"{threshold}°F" if threshold else "Unknown"
            
            # Get token IDs (Polymarket has YES and NO tokens)
            tokens = market.get('tokens', [])
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
                market_info.append({
                    'question': question,
                    'threshold': threshold,
                    'threshold_display': threshold_display,
                    'market_id': market.get('id'),
                    'condition_id': market.get('conditionId'),
                    'yes_token_id': token_ids[0],  # First token is YES
                    'no_token_id': token_ids[1] if len(token_ids) > 1 else None,
                })
        
        return market_info
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_price_history(token_id, interval='1d', fidelity=5):
    """
    Fetch historical price data for a token.
    
    Args:
        token_id: The CLOB token ID
        interval: Time interval (1m, 1h, 6h, 1d, 1w, max)
        fidelity: Resolution in minutes (default: 5 for 5-minute intervals)
    
    Returns:
        DataFrame with timestamp and price columns
    """
    url = "https://clob.polymarket.com/prices-history"
    
    params = {
        'market': token_id,
        'interval': interval,
        'fidelity': fidelity
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        history = data.get('history', [])
        
        if not history:
            return pd.DataFrame()
        
        # Convert to DataFrame
        records = []
        for point in history:
            timestamp = datetime.fromtimestamp(point['t'], tz=NY_TZ)
            price = point['p']
            
            records.append({
                'timestamp': timestamp,
                'price': price,
                'probability': price  # Price is probability in Polymarket
            })
        
        df = pd.DataFrame(records)
        return df
        
    except Exception as e:
        print(f"  Warning: Could not fetch history for token {token_id[:8]}...: {e}")
        return pd.DataFrame()


def fetch_all_market_histories(event_slug, interval='1d', fidelity=5):
    """
    Fetch historical odds for all markets in an event.
    
    Args:
        event_slug: The event identifier
        interval: Time interval (1m, 1h, 6h, 1d, 1w, max)
        fidelity: Resolution in minutes
    
    Returns:
        Dict mapping market questions to DataFrames with historical odds
    """
    # Get market info
    markets = fetch_market_token_ids(event_slug)
    
    if not markets:
        return None
    
    print(f"\nFetching historical odds (interval={interval}, fidelity={fidelity} min)...")
    
    all_histories = {}
    
    for market in markets:
        question = market['question']
        threshold = market['threshold']
        threshold_display = market['threshold_display']
        token_id = market['yes_token_id']
        
        print(f"  Fetching: {threshold_display}...")
        
        df = fetch_price_history(token_id, interval=interval, fidelity=fidelity)
        
        if not df.empty:
            df['threshold'] = threshold
            df['threshold_display'] = threshold_display
            df['question'] = question
            all_histories[threshold] = df
            print(f"    ✓ Got {len(df)} data points")
        else:
            print(f"    ✗ No data")
    
    return all_histories


def save_historical_odds(histories, output_file='data/raw/polymarket_odds_history.csv'):
    """Save all historical odds to a single CSV file."""
    if not histories:
        print("No data to save")
        return
    
    # Combine all DataFrames
    all_dfs = []
    for threshold, df in histories.items():
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(['threshold', 'timestamp'])
    
    combined_df.to_csv(output_file, index=False)
    print(f"\n✓ Historical odds saved to: {output_file}")
    print(f"  Total records: {len(combined_df)}")
    print(f"  Time range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    return combined_df


def display_latest_odds(histories):
    """Display the most recent odds for each market."""
    if not histories:
        return
    
    print("\n" + "=" * 70)
    print("Latest Odds (Most Recent Data Point)")
    print("=" * 70)
    print(f"{'Range':<15} {'Latest Odds':<15} {'Time':<25}")
    print("-" * 70)
    
    # Sort by threshold (handle different formats)
    def sort_key(t):
        if isinstance(t, str):
            if '-' in t:
                return int(t.split('-')[0])
            elif '≤' in t:
                return int(t.replace('≤', ''))
            elif '≥' in t:
                return int(t.replace('≥', '')) + 100  # Put "or higher" at end
        return 0
    
    for threshold in sorted(histories.keys(), key=sort_key):
        df = histories[threshold]
        if not df.empty:
            latest = df.iloc[-1]
            display = latest.get('threshold_display', threshold)
            print(f"{display:<15} {latest['probability']:.1%}{'':<7} {latest['timestamp']}")
    
    print("-" * 70)


def get_event_slug_for_date(target_date=None):
    """
    Generate event slug for a specific date.
    
    Args:
        target_date: datetime object (default: tomorrow)
    
    Returns:
        Event slug string
    """
    if target_date is None:
        target_date = datetime.now() + timedelta(days=1)
    
    month = target_date.strftime('%B').lower()
    day = target_date.day
    
    slug = f"highest-temperature-in-nyc-on-{month}-{day}"
    
    return slug


if __name__ == "__main__":
    print("=" * 70)
    print("Polymarket Historical Odds Fetcher")
    print("=" * 70)
    print()
    
    # Get tomorrow's event
    tomorrow = datetime.now() + timedelta(days=1)
    slug = get_event_slug_for_date(tomorrow)
    
    print(f"Event date: {tomorrow.strftime('%B %d, %Y')}")
    print(f"Event slug: {slug}\n")
    
    # Fetch historical odds
    # interval options: 1m, 1h, 6h, 1d, 1w, max
    # fidelity: resolution in minutes (5 = 5-minute intervals)
    histories = fetch_all_market_histories(
        event_slug=slug,
        interval='1d',  # Last 24 hours
        fidelity=5      # 5-minute resolution
    )
    
    if histories:
        display_latest_odds(histories)
        df = save_historical_odds(histories)
        
        print("\n" + "=" * 70)
        print("Success! Historical odds retrieved.")
        print("=" * 70)
        
        # Show sample of data
        print("\nSample data (first 5 rows):")
        print(df.head())
    else:
        print("\n" + "=" * 70)
        print("Could not fetch historical data.")
        print("The market may not exist yet or may not have trading history.")
        print("=" * 70)
