"""Debug Polymarket API response"""

from datetime import date, timedelta
import requests
import json

target_date = date.today() + timedelta(days=1)
month = target_date.strftime('%B').lower()
day = target_date.day
slug = f"highest-temperature-in-nyc-on-{month}-{day}"

api_url = f"https://gamma-api.polymarket.com/events?slug={slug}"
headers = {
    'User-Agent': 'Mozilla/5.0',
    'Accept': 'application/json'
}

response = requests.get(api_url, headers=headers, timeout=10)
data = response.json()

if data and len(data) > 0:
    event = data[0]
    print(f"Event: {event.get('title', 'N/A')}")
    print(f"Markets: {len(event.get('markets', []))}")
    print()
    
    for i, market in enumerate(event.get('markets', [])[:2]):
        print(f"Market {i+1}:")
        print(f"  Question: {market.get('question', 'N/A')}")
        print(f"  outcomePrices type: {type(market.get('outcomePrices'))}")
        print(f"  outcomePrices value: {market.get('outcomePrices')}")
        print(f"  volume: {market.get('volume')}")
        print()
        
        # Try to find the actual price field
        print("  All market fields:")
        for key in market.keys():
            if 'price' in key.lower() or 'prob' in key.lower():
                print(f"    {key}: {market[key]}")
        print()
