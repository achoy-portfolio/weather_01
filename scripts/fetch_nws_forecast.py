"""
Fetch NWS weather forecast for KLGA (LaGuardia Airport).
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import json

def get_nws_forecast(lat=40.7769, lon=-73.8740):
    """
    Fetch NWS forecast for given coordinates (default: KLGA).
    
    Returns:
        DataFrame with hourly forecast data
    """
    
    # Step 1: Get the forecast grid endpoint for this location
    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    
    headers = {
        'User-Agent': '(Weather Analysis, contact@example.com)',
        'Accept': 'application/json'
    }
    
    print(f"Fetching forecast for coordinates: {lat}, {lon}")
    
    try:
        # Get grid point
        response = requests.get(points_url, headers=headers, timeout=10)
        response.raise_for_status()
        points_data = response.json()
        
        forecast_hourly_url = points_data['properties']['forecastHourly']
        print(f"✓ Grid point found")
        
        # Get hourly forecast
        response = requests.get(forecast_hourly_url, headers=headers, timeout=10)
        response.raise_for_status()
        forecast_data = response.json()
        
        # Parse forecast periods
        periods = forecast_data['properties']['periods']
        print(f"✓ Retrieved {len(periods)} hourly forecast periods")
        
        # Convert to DataFrame
        forecast_list = []
        for period in periods:
            forecast_list.append({
                'start_time': period['startTime'],
                'end_time': period['endTime'],
                'temperature': period['temperature'],
                'temperature_unit': period['temperatureUnit'],
                'wind_speed': period['windSpeed'],
                'wind_direction': period['windDirection'],
                'short_forecast': period['shortForecast'],
                'detailed_forecast': period.get('detailedForecast', ''),
                'precipitation_probability': period.get('probabilityOfPrecipitation', {}).get('value', None),
                'dewpoint': period.get('dewpoint', {}).get('value', None),
                'relative_humidity': period.get('relativeHumidity', {}).get('value', None)
            })
        
        df = pd.DataFrame(forecast_list)
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        # Extract numeric wind speed (handle "5 to 10 mph" format)
        df['wind_speed_mph'] = df['wind_speed'].str.extract(r'(\d+)').astype(float)
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching forecast: {e}")
        return None
    except Exception as e:
        print(f"✗ Error processing forecast: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_daily_forecast_summary(df):
    """Aggregate hourly forecast into daily summaries."""
    if df is None or len(df) == 0:
        return None
    
    df['date'] = df['start_time'].dt.date
    
    daily = df.groupby('date').agg({
        'temperature': ['min', 'max', 'mean'],
        'wind_speed_mph': ['mean', 'max'],
        'precipitation_probability': 'max',
        'dewpoint': 'mean',
        'relative_humidity': 'mean'
    }).reset_index()
    
    daily.columns = ['date', 'temp_min_forecast', 'temp_max_forecast', 'temp_mean_forecast',
                     'wind_mean_forecast', 'wind_max_forecast', 'precip_prob_max',
                     'dewpoint_forecast', 'humidity_forecast']
    
    return daily

def save_forecast(output_file='data/raw/nws_forecast_klga.csv'):
    """Fetch and save current NWS forecast."""
    
    print("=" * 70)
    print("NWS Forecast Fetcher - KLGA")
    print("=" * 70)
    print()
    
    # Fetch hourly forecast
    df_hourly = get_nws_forecast()
    
    if df_hourly is not None:
        # Save hourly
        df_hourly.to_csv(output_file, index=False)
        print(f"\n✓ Hourly forecast saved: {output_file}")
        
        # Show daily summary
        df_daily = get_daily_forecast_summary(df_hourly)
        
        print("\nDaily Forecast Summary:")
        print(df_daily.to_string(index=False))
        
        # Save daily summary
        daily_file = output_file.replace('.csv', '_daily.csv')
        df_daily.to_csv(daily_file, index=False)
        print(f"\n✓ Daily summary saved: {daily_file}")
        
        return df_hourly, df_daily
    
    return None, None

if __name__ == "__main__":
    hourly, daily = save_forecast()
    
    if hourly is not None:
        print("\n" + "=" * 70)
        print("Forecast Retrieved Successfully!")
        print("=" * 70)
