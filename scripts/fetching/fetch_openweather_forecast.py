"""
Fetch weather forecast from OpenWeatherMap API for KLGA.
Free tier: 1000 calls/day
Sign up: https://openweathermap.org/api
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# KLGA coordinates
KLGA_LAT = 40.7769
KLGA_LON = -73.8740

def get_openweather_forecast(lat=KLGA_LAT, lon=KLGA_LON, api_key=None):
    """
    Fetch OpenWeatherMap forecast.
    
    Args:
        lat: Latitude (default: KLGA)
        lon: Longitude (default: KLGA)
        api_key: OpenWeatherMap API key (or set OPENWEATHER_API_KEY env var)
    
    Returns:
        DataFrame with hourly forecast
    """
    
    # Get API key
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("✗ No API key provided")
        print("  Get free key at: https://openweathermap.org/api")
        print("  Set environment variable: OPENWEATHER_API_KEY=your_key")
        print("  Or pass as parameter: get_openweather_forecast(api_key='your_key')")
        return None
    
    print(f"Fetching OpenWeatherMap forecast for ({lat}, {lon})...")
    
    # API endpoint (One Call API 3.0)
    url = "https://api.openweathermap.org/data/3.0/onecall"
    
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'imperial',  # Fahrenheit
        'exclude': 'minutely,alerts'  # We want hourly and daily
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"✓ Forecast retrieved")
        
        # Parse hourly forecast
        hourly_list = []
        for hour in data.get('hourly', [])[:48]:  # Next 48 hours
            hourly_list.append({
                'timestamp': datetime.fromtimestamp(hour['dt']),
                'temperature': hour['temp'],
                'feels_like': hour['feels_like'],
                'humidity': hour['humidity'],
                'wind_speed': hour['wind_speed'],
                'wind_gust': hour.get('wind_gust', 0),
                'precipitation_prob': hour.get('pop', 0) * 100,
                'description': hour['weather'][0]['description']
            })
        
        hourly_df = pd.DataFrame(hourly_list)
        
        # Parse daily forecast
        daily_list = []
        for day in data.get('daily', [])[:7]:  # Next 7 days
            daily_list.append({
                'date': datetime.fromtimestamp(day['dt']).date(),
                'temp_min': day['temp']['min'],
                'temp_max': day['temp']['max'],
                'temp_morning': day['temp']['morn'],
                'temp_day': day['temp']['day'],
                'temp_evening': day['temp']['eve'],
                'temp_night': day['temp']['night'],
                'humidity': day['humidity'],
                'wind_speed': day['wind_speed'],
                'precipitation_prob': day.get('pop', 0) * 100,
                'description': day['weather'][0]['description']
            })
        
        daily_df = pd.DataFrame(daily_list)
        
        print(f"✓ Hourly: {len(hourly_df)} periods")
        print(f"✓ Daily: {len(daily_df)} days")
        
        return hourly_df, daily_df
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            print("✗ Invalid API key")
            print("  Get free key at: https://openweathermap.org/api")
        else:
            print(f"✗ HTTP Error: {e}")
        return None, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None

def get_tomorrow_forecast(api_key=None):
    """Get tomorrow's forecast high."""
    hourly_df, daily_df = get_openweather_forecast(api_key=api_key)
    
    if daily_df is not None and len(daily_df) > 0:
        tomorrow = datetime.now().date() + timedelta(days=1)
        tomorrow_forecast = daily_df[daily_df['date'] == tomorrow]
        
        if len(tomorrow_forecast) > 0:
            high = tomorrow_forecast['temp_max'].iloc[0]
            low = tomorrow_forecast['temp_min'].iloc[0]
            
            print(f"\nTomorrow's Forecast:")
            print(f"  High: {high:.1f}°F")
            print(f"  Low: {low:.1f}°F")
            
            return high, low
    
    return None, None

if __name__ == "__main__":
    print("=" * 70)
    print("OpenWeatherMap Forecast Fetcher")
    print("=" * 70)
    print()
    
    # Check for API key
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        print("⚠ No API key found")
        print("\nSetup Instructions:")
        print("1. Sign up at: https://openweathermap.org/api")
        print("2. Get your free API key")
        print("3. Set environment variable:")
        print("   Windows: $env:OPENWEATHER_API_KEY='your_key'")
        print("   Linux/Mac: export OPENWEATHER_API_KEY='your_key'")
        print("\nOr edit this script and add: api_key='your_key'")
        exit(1)
    
    # Fetch forecast
    hourly_df, daily_df = get_openweather_forecast()
    
    if hourly_df is not None:
        print("\n" + "=" * 70)
        print("Tomorrow's Forecast:")
        print("=" * 70)
        
        high, low = get_tomorrow_forecast()
        
        # Save to files
        hourly_df.to_csv('data/raw/openweather_hourly.csv', index=False)
        daily_df.to_csv('data/raw/openweather_daily.csv', index=False)
        
        print(f"\n✓ Saved to:")
        print(f"  - data/raw/openweather_hourly.csv")
        print(f"  - data/raw/openweather_daily.csv")
