"""
Fetch current odds from Polymarket temperature markets.
"""

import requests
import json
from datetime import datetime
import pandas as pd

def fetch_polymarket_event(event_slug):
    """
    Fetch Polymarket event data including all markets and current odds.
    
    Args:
        event_slug: The event identifier from URL (e.g., 'highest-temperature-in-nyc-on-january-17')
    
    Returns:
        Dict with event data and market odds
    """
    
    # Polymarket API endpoint
    api_url = f"https://gamma-api.polymarket.com/events?slug={event_slug}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    print(f"Fetching Polymarket event: {event_slug}")
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or len(data) == 0:
            print("✗ Event not found")
            return None
        
        event = data[0]
        
        print(f"✓ Event found: {event.get('title', 'Unknown')}")
        
        # Extract markets (different temperature thresholds)
        markets = event.get('markets', [])
        
        if not markets:
            print("✗ No markets found in event")
            return None
        
        print(f"✓ Found {len(markets)} markets")
        
        # Parse market data
        market_data = []
        for market in markets:
            # Debug: print first market structure
            if len(market_data) == 0:
                print(f"\nDebug - First market:")
                print(f"  Question: {market.get('question', '')}")
                print(f"  outcomePrices: {market.get('outcomePrices')}")
                print(f"  outcomes: {market.get('outcomes')}")
                print(f"  lastTradePrice: {market.get('lastTradePrice')}")
            
            # Extract temperature threshold from question
            question = market.get('question', '')
            
            # Try to extract threshold (e.g., "Will the temperature be 35°F or higher?")
            import re
            threshold_match = re.search(r'(\d+)°F', question)
            threshold = int(threshold_match.group(1)) if threshold_match else None
            
            # Determine if it's "or higher" or "or below"
            is_above = 'or higher' in question.lower() or 'or above' in question.lower()
            is_below = 'or below' in question.lower() or 'or lower' in question.lower()
            
            # Get current price (probability)
            # Polymarket stores outcomes as JSON string and prices in outcomePrices
            price = None
            
            # Try outcomePrices first (should be a list like ["0.0045", "0.9955"])
            outcome_prices = market.get('outcomePrices')
            if outcome_prices:
                try:
                    # Parse if it's a string
                    if isinstance(outcome_prices, str):
                        import json
                        outcome_prices = json.loads(outcome_prices)
                    
                    # outcomePrices is [Yes, No]
                    if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                        yes_price = float(outcome_prices[0])
                        no_price = float(outcome_prices[1])
                        
                        # Always store the Yes price (probability of the stated outcome)
                        price = yes_price
                                
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    print(f"  Warning: Could not parse price for {question[:30]}: {e}")
            
            # Fallback: try lastTradePrice
            if price is None:
                last_trade = market.get('lastTradePrice')
                if last_trade:
                    try:
                        price = float(last_trade)
                    except (ValueError, TypeError):
                        pass
            
            market_data.append({
                'question': question,
                'threshold': threshold,
                'market_id': market.get('id'),
                'condition_id': market.get('conditionId'),
                'yes_price': price,
                'yes_probability': price,  # Price is probability in Polymarket
                'volume': float(market.get('volume', 0)) if market.get('volume') else 0,
                'liquidity': float(market.get('liquidity', 0)) if market.get('liquidity') else 0,
                'direction': 'above' if is_above else ('below' if is_below else 'unknown')
            })
        
        # Sort by threshold
        market_data = sorted(market_data, key=lambda x: x['threshold'] if x['threshold'] else 0)
        
        return {
            'event': event,
            'markets': market_data,
            'fetch_time': datetime.now().isoformat()
        }
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"✗ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_market_odds(data):
    """Display market odds in a readable format."""
    if not data:
        return
    
    event = data['event']
    markets = data['markets']
    
    print("\n" + "=" * 70)
    print(f"Event: {event.get('title', 'Unknown')}")
    print(f"Date: {event.get('endDate', 'Unknown')}")
    print("=" * 70)
    
    print("\nCurrent Market Odds:")
    print("-" * 70)
    print(f"{'Threshold':<15} {'Question':<35} {'Odds':<10} {'Volume':<15}")
    print("-" * 70)
    
    for market in markets:
        threshold = f"{market['threshold']}°F" if market['threshold'] else "N/A"
        question = market['question'][:35]
        odds = f"{market['yes_probability']:.1%}" if market['yes_probability'] else "N/A"
        volume = f"${market['volume']:,.0f}" if market['volume'] else "$0"
        
        print(f"{threshold:<15} {question:<35} {odds:<10} {volume:<15}")
    
    print("-" * 70)

def save_market_data(data, output_file='data/raw/polymarket_odds.json'):
    """Save market data to file."""
    if not data:
        return
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Market data saved to: {output_file}")
    
    # Also save as CSV for easy analysis
    csv_file = output_file.replace('.json', '.csv')
    df = pd.DataFrame(data['markets'])
    df.to_csv(csv_file, index=False)
    print(f"✓ Market data saved to: {csv_file}")

def get_todays_event_slug():
    """
    Generate event slug for today or tomorrow.
    Format: highest-temperature-in-nyc-on-january-17
    """
    from datetime import datetime, timedelta
    
    # Check tomorrow (markets are usually for next day)
    tomorrow = datetime.now() + timedelta(days=1)
    
    month = tomorrow.strftime('%B').lower()
    day = tomorrow.day
    
    slug = f"highest-temperature-in-nyc-on-{month}-{day}"
    
    return slug

if __name__ == "__main__":
    print("=" * 70)
    print("Polymarket Temperature Odds Fetcher")
    print("=" * 70)
    print()
    
    # Try to fetch tomorrow's market
    slug = get_todays_event_slug()
    print(f"Attempting to fetch tomorrow's market: {slug}\n")
    
    data = fetch_polymarket_event(slug)
    
    if data:
        display_market_odds(data)
        save_market_data(data)
        
        print("\n" + "=" * 70)
        print("Success! Market odds retrieved.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("Could not fetch market data.")
        print("Try manually specifying the event slug from the Polymarket URL.")
        print("=" * 70)
        print("\nExample usage:")
        print("  from fetch_polymarket_odds import fetch_polymarket_event")
        print("  data = fetch_polymarket_event('highest-temperature-in-nyc-on-january-17')")
