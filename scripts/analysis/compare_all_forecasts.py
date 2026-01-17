"""
Compare weather forecasts from multiple free APIs.
Gets tomorrow's high temperature from:
- NWS (National Weather Service)
- Open-Meteo
- Weather.gov
- 7Timer
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import statistics

# KLGA coordinates
KLGA_LAT = 40.7769
KLGA_LON = -73.8740

def get_nws_forecast():
    """Fetch NWS forecast."""
    try:
        from fetch_nws_forecast import get_nws_forecast, get_daily_forecast_summary
        
        forecast_df = get_nws_forecast()
        if forecast_df is not None:
            daily = get_daily_forecast_summary(forecast_df)
            tomorrow = datetime.now().date() + timedelta(days=1)
            tomorrow_fc = daily[daily['date'] == tomorrow]
            
            if len(tomorrow_fc) > 0:
                high = tomorrow_fc['temp_max_forecast'].iloc[0]
                low = tomorrow_fc['temp_min_forecast'].iloc[0]
                return {'high': high, 'low': low, 'source': 'NWS'}
    except Exception as e:
        print(f"  NWS error: {e}")
    
    return None

def get_openmeteo_forecast():
    """Fetch Open-Meteo forecast."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': KLGA_LAT,
            'longitude': KLGA_LON,
            'daily': 'temperature_2m_max,temperature_2m_min',
            'temperature_unit': 'fahrenheit',
            'timezone': 'America/New_York'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Find tomorrow's date
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        dates = data['daily']['time']
        
        if tomorrow in dates:
            idx = dates.index(tomorrow)
            high = data['daily']['temperature_2m_max'][idx]
            low = data['daily']['temperature_2m_min'][idx]
        else:
            # Fallback to index 1
            high = data['daily']['temperature_2m_max'][1]
            low = data['daily']['temperature_2m_min'][1]
        
        return {'high': high, 'low': low, 'source': 'Open-Meteo'}
    except Exception as e:
        print(f"  Open-Meteo error: {e}")
    
    return None

def get_7timer_forecast():
    """Fetch 7Timer forecast (free, no key)."""
    try:
        url = "https://www.7timer.info/bin/api.pl"
        params = {
            'lon': KLGA_LON,
            'lat': KLGA_LAT,
            'product': 'civil',
            'output': 'json'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Find tomorrow's forecast (next 24-48 hours)
        tomorrow_temps = []
        for period in data['dataseries'][:8]:  # Next 8 periods (3-hour intervals)
            temp_c = period['temp2m']
            temp_f = temp_c * 9/5 + 32
            tomorrow_temps.append(temp_f)
        
        if tomorrow_temps:
            high = max(tomorrow_temps)
            low = min(tomorrow_temps)
            return {'high': high, 'low': low, 'source': '7Timer'}
    except Exception as e:
        print(f"  7Timer error: {e}")
    
    return None

def get_weatherapi_forecast():
    """Fetch WeatherAPI.com forecast (free tier: 1M calls/month)."""
    try:
        # Note: Requires API key from https://www.weatherapi.com/
        # Free tier is very generous
        import os
        api_key = os.getenv('WEATHERAPI_KEY')
        
        if not api_key:
            print(f"  WeatherAPI: No API key (optional)")
            return None
        
        url = "http://api.weatherapi.com/v1/forecast.json"
        params = {
            'key': api_key,
            'q': f"{KLGA_LAT},{KLGA_LON}",
            'days': 2
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Tomorrow is index 1
        tomorrow = data['forecast']['forecastday'][1]['day']
        high = tomorrow['maxtemp_f']
        low = tomorrow['mintemp_f']
        
        return {'high': high, 'low': low, 'source': 'WeatherAPI'}
    except Exception as e:
        print(f"  WeatherAPI error: {e}")
    
    return None

def get_wttr_forecast():
    """Fetch wttr.in forecast (free, no key)."""
    try:
        # Simple weather service
        url = f"https://wttr.in/{KLGA_LAT},{KLGA_LON}"
        params = {'format': 'j1'}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Find tomorrow's date
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Search for tomorrow in the weather array
        for day in data['weather']:
            if day['date'] == tomorrow:
                high = int(day['maxtempF'])
                low = int(day['mintempF'])
                return {'high': high, 'low': low, 'source': 'wttr.in'}
        
        # Fallback to index 0 if tomorrow not found
        tomorrow_data = data['weather'][0]
        high = int(tomorrow_data['maxtempF'])
        low = int(tomorrow_data['mintempF'])
        
        return {'high': high, 'low': low, 'source': 'wttr.in'}
    except Exception as e:
        print(f"  wttr.in error: {e}")
    
    return None

def compare_all_forecasts():
    """Fetch and compare all forecasts."""
    
    print("=" * 70)
    print("MULTI-SOURCE WEATHER FORECAST COMPARISON")
    print("=" * 70)
    print(f"Location: KLGA ({KLGA_LAT}, {KLGA_LON})")
    print(f"Date: Tomorrow ({(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')})")
    print("=" * 70)
    print()
    
    forecasts = []
    
    # Fetch from all sources
    print("Fetching forecasts...")
    
    sources = [
        ('NWS', get_nws_forecast),
        ('Open-Meteo', get_openmeteo_forecast),
        ('wttr.in', get_wttr_forecast),
        # ('7Timer', get_7timer_forecast),  # Disabled: often outdated/inaccurate
        ('WeatherAPI', get_weatherapi_forecast)
    ]
    
    for name, func in sources:
        print(f"  {name}...", end=' ')
        result = func()
        if result:
            forecasts.append(result)
            print(f"✓ High: {result['high']:.1f}°F, Low: {result['low']:.1f}°F")
        else:
            print("✗ Failed")
    
    if not forecasts:
        print("\n✗ No forecasts retrieved")
        return None
    
    # Create comparison table
    print("\n" + "=" * 70)
    print("FORECAST COMPARISON")
    print("=" * 70)
    print(f"{'Source':<20} {'High':<10} {'Low':<10} {'Range':<10}")
    print("-" * 70)
    
    highs = []
    lows = []
    
    for fc in forecasts:
        range_val = fc['high'] - fc['low']
        print(f"{fc['source']:<20} {fc['high']:>6.1f}°F   {fc['low']:>6.1f}°F   {range_val:>6.1f}°F")
        highs.append(float(fc['high']))  # Convert to Python float
        lows.append(float(fc['low']))    # Convert to Python float
    
    print("-" * 70)
    
    # Calculate consensus
    avg_high = statistics.mean(highs)
    avg_low = statistics.mean(lows)
    median_high = statistics.median(highs)
    median_low = statistics.median(lows)
    std_high = statistics.stdev(highs) if len(highs) > 1 else 0
    std_low = statistics.stdev(lows) if len(lows) > 1 else 0
    
    print(f"{'AVERAGE':<20} {avg_high:>6.1f}°F   {avg_low:>6.1f}°F")
    print(f"{'MEDIAN':<20} {median_high:>6.1f}°F   {median_low:>6.1f}°F")
    print(f"{'STD DEV':<20} {std_high:>6.1f}°F   {std_low:>6.1f}°F")
    print(f"{'RANGE':<20} {min(highs):.1f}-{max(highs):.1f}°F {min(lows):.1f}-{max(lows):.1f}°F")
    
    print("\n" + "=" * 70)
    print("CONSENSUS FORECAST")
    print("=" * 70)
    print(f"High: {median_high:.1f}°F ± {std_high:.1f}°F")
    print(f"Low:  {median_low:.1f}°F ± {std_low:.1f}°F")
    print(f"\nBased on {len(forecasts)} sources")
    
    # Uncertainty assessment
    if std_high > 3:
        print("\n⚠ High uncertainty - forecasts disagree significantly")
    elif std_high > 1.5:
        print("\n⚠ Moderate uncertainty - some disagreement between forecasts")
    else:
        print("\n✓ Low uncertainty - forecasts agree well")
    
    # Save results
    df = pd.DataFrame(forecasts)
    df.to_csv('data/raw/forecast_comparison.csv', index=False)
    print(f"\n✓ Saved to: data/raw/forecast_comparison.csv")
    
    return {
        'consensus_high': median_high,
        'consensus_low': median_low,
        'uncertainty_high': std_high,
        'uncertainty_low': std_low,
        'num_sources': len(forecasts),
        'forecasts': forecasts
    }

if __name__ == "__main__":
    result = compare_all_forecasts()
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print("\nTip: Use the median forecast for your model")
    print("     Higher std dev = less confident prediction")
