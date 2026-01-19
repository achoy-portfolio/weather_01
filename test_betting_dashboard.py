"""Test betting recommendation dashboard components"""

from datetime import date, timedelta
import requests
import json

# Test Open-Meteo forecast
def test_openmeteo():
    target_date = date.today() + timedelta(days=1)
    url = "https://api.open-meteo.com/v1/forecast"
    
    lat = 40.7769
    lon = -73.8740
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m',
        'temperature_unit': 'fahrenheit',
        'timezone': 'America/New_York',
        'forecast_days': 3
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'hourly' in data:
            times = data['hourly'].get('time', [])
            temps = data['hourly'].get('temperature_2m', [])
            print(f"✅ Open-Meteo: Got {len(times)} hourly forecasts")
            
            # Find target date temps
            target_temps = []
            for time_str, temp in zip(times, temps):
                if temp is not None and target_date.isoformat() in time_str:
                    target_temps.append(temp)
            
            if target_temps:
                print(f"   Target date ({target_date}): {len(target_temps)} hours, max {max(target_temps):.1f}°F")
            else:
                print(f"   ⚠️ No data for target date {target_date}")
        else:
            print("❌ No hourly data in response")
            
    except Exception as e:
        print(f"❌ Open-Meteo error: {e}")

# Test Polymarket API
def test_polymarket():
    target_date = date.today() + timedelta(days=1)
    month = target_date.strftime('%B').lower()
    day = target_date.day
    slug = f"highest-temperature-in-nyc-on-{month}-{day}"
    
    api_url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            event = data[0]
            markets = event.get('markets', [])
            print(f"✅ Polymarket: Found {len(markets)} markets for {target_date}")
            
            for market in markets[:3]:  # Show first 3
                question = market.get('question', '')
                outcome_prices = market.get('outcomePrices', [])
                
                # Handle JSON string or array
                if isinstance(outcome_prices, str):
                    try:
                        outcome_prices = json.loads(outcome_prices)
                    except:
                        outcome_prices = []
                
                if outcome_prices and len(outcome_prices) > 0:
                    try:
                        price = float(outcome_prices[0])
                        print(f"   {question[:50]}... = {price:.1%}")
                    except:
                        print(f"   ⚠️ Could not parse price: {outcome_prices[0]}")
        else:
            print(f"⚠️ No Polymarket market found for {target_date}")
            print(f"   Tried slug: {slug}")
            
    except Exception as e:
        print(f"❌ Polymarket error: {e}")

if __name__ == '__main__':
    print("Testing Betting Dashboard Components\n")
    print("=" * 50)
    test_openmeteo()
    print()
    test_polymarket()
    print("=" * 50)
