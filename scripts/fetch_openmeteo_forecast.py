"""
Fetch weather forecast from Open-Meteo API for KLGA.
Completely free, no API key required!
Docs: https://open-meteo.com/en/docs
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

# KLGA coordinates
KLGA_LAT = 40.7769
KLGA_LON = -73.8740

def get_openmeteo_forecast(lat=KLGA_LAT, lon=KLGA_LON):
    """
    Fetch Open-Meteo forecast.
    
    Args:
        lat: Latitude (default: KLGA)
        lon: Longitude (default: KLGA)
    
    Returns:
        hourly_df, daily_df: DataFrames with forecast data
    """
    
    print(f"Fetching Open-Meteo forecast for ({lat}, {lon})...")
    
    # API endpoint
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,relative_humidity_2m,precipitation_probability,wind_speed_10m,wind_gusts_10m',
        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max',
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'precipitation_unit': 'inch',
        'timezone': 'America/New_York'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"✓ Forecast retrieved")
        
        # Parse hourly data
        hourly = data['hourly']
        hourly_df = pd.DataFrame({
            'timestamp': pd.to_datetime(hourly['time']),
            'temperature': hourly['temperature_2m'],
            'humidity': hourly['relative_humidity_2m'],
            'precipitation_prob': hourly['precipitation_probability'],
            'wind_speed': hourly['wind_speed_10m'],
            'wind_gust': hourly['wind_gusts_10m']
        })
        
        # Parse daily data
        daily = data['daily']
        daily_df = pd.DataFrame({
            'date': pd.to_datetime(daily['time']),
            'temp_max': daily['temperature_2m_max'],
            'temp_min': daily['temperature_2m_min'],
            'precipitation_prob': daily['precipitation_probability_max'],
            'wind_speed_max': daily['wind_speed_10m_max']
        })
        
        # Convert date column to date objects
        daily_df['date'] = daily_df['date'].dt.date
        
        print(f"✓ Hourly: {len(hourly_df)} periods")
        print(f"✓ Daily: {len(daily_df)} days")
        
        return hourly_df, daily_df
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None

def get_tomorrow_forecast():
    """Get tomorrow's forecast high."""
    hourly_df, daily_df = get_openmeteo_forecast()
    
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

def compare_with_nws():
    """Compare Open-Meteo with NWS forecast."""
    
    print("=" * 70)
    print("Open-Meteo vs NWS Forecast Comparison")
    print("=" * 70)
    
    # Get Open-Meteo
    om_high, om_low = get_tomorrow_forecast()
    
    # Get NWS
    try:
        from fetch_nws_forecast import get_nws_forecast, get_daily_forecast_summary
        
        nws_df = get_nws_forecast()
        if nws_df is not None:
            nws_daily = get_daily_forecast_summary(nws_df)
            tomorrow = datetime.now().date() + timedelta(days=1)
            nws_tomorrow = nws_daily[nws_daily['date'] == tomorrow]
            
            if len(nws_tomorrow) > 0:
                nws_high = nws_tomorrow['temp_max_forecast'].iloc[0]
                nws_low = nws_tomorrow['temp_min_forecast'].iloc[0]
                
                print(f"\nNWS Forecast:")
                print(f"  High: {nws_high:.1f}°F")
                print(f"  Low: {nws_low:.1f}°F")
                
                if om_high and nws_high:
                    diff_high = om_high - nws_high
                    diff_low = om_low - nws_low
                    print(f"\nDifference:")
                    print(f"  High: {diff_high:+.1f}°F")
                    print(f"  Low: {diff_low:+.1f}°F")
    except Exception as e:
        print(f"Could not fetch NWS: {e}")

if __name__ == "__main__":
    print("=" * 70)
    print("Open-Meteo Forecast Fetcher")
    print("=" * 70)
    print("Free, no API key required!")
    print()
    
    # Fetch forecast
    hourly_df, daily_df = get_openmeteo_forecast()
    
    if hourly_df is not None:
        print("\n" + "=" * 70)
        
        # Get tomorrow's forecast
        high, low = get_tomorrow_forecast()
        
        # Save to files
        hourly_df.to_csv('data/raw/openmeteo_hourly.csv', index=False)
        daily_df.to_csv('data/raw/openmeteo_daily.csv', index=False)
        
        print(f"\n✓ Saved to:")
        print(f"  - data/raw/openmeteo_hourly.csv")
        print(f"  - data/raw/openmeteo_daily.csv")
        
        # Compare with NWS
        print("\n" + "=" * 70)
        compare_with_nws()
        
        print("\n" + "=" * 70)
        print("Success!")
        print("=" * 70)
