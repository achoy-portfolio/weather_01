"""
Fetch WeatherNext forecast data from Google Earth Engine for KLGA.
Requires: pip install earthengine-api
Setup: https://developers.google.com/earth-engine/guides/python_install
"""

import ee
import pandas as pd
from datetime import datetime, timedelta

# KLGA coordinates
KLGA_LAT = 40.7769
KLGA_LON = -73.8740

def initialize_earth_engine(project='ee-project-484606'):
    """Initialize Earth Engine with authentication."""
    try:
        ee.Initialize(project=project)
        print(f"✓ Earth Engine initialized with project: {project}")
        return True
    except Exception as e:
        print(f"⚠ Earth Engine not initialized: {e}")
        print("\n  To fix this:")
        print(f"  1. Make sure project '{project}' has Earth Engine API enabled")
        print("  2. Go to: https://console.cloud.google.com/apis/library/earthengine.googleapis.com")
        print("  3. Select your project and click 'Enable'")
        print(f"  4. Or try: ee.Initialize(project='{project}')")
        return False

def get_weathernext_forecast(lat=KLGA_LAT, lon=KLGA_LON, days_ahead=7):
    """
    Fetch WeatherNext forecast for a location.
    
    Args:
        lat: Latitude (default: KLGA)
        lon: Longitude (default: KLGA)
        days_ahead: Number of days to forecast (default: 7)
    
    Returns:
        DataFrame with forecast data
    """
    
    if not initialize_earth_engine():
        return None
    
    print(f"\nFetching WeatherNext forecast for ({lat}, {lon})...")
    
    try:
        # WeatherNext dataset
        dataset_id = 'projects/gcp-public-data-weathernext/assets/126478713_1_0'
        
        # Create point geometry
        point = ee.Geometry.Point([lon, lat])
        
        # Get the image collection
        collection = ee.ImageCollection(dataset_id)
        
        # Filter to recent forecasts
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        filtered = collection.filterDate(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Get the most recent forecast
        latest = filtered.sort('system:time_start', False).first()
        
        if latest is None:
            print("✗ No forecast data available")
            return None
        
        # Extract forecast bands
        # WeatherNext typically has bands like:
        # - temperature_2m
        # - precipitation
        # - wind_speed_10m
        # - relative_humidity_2m
        
        # Sample the forecast at the point
        forecast_info = latest.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=1000  # 1km resolution
        ).getInfo()
        
        print(f"✓ Forecast retrieved")
        print(f"  Available bands: {list(forecast_info.keys())}")
        
        # Parse the forecast data
        forecast_data = {
            'timestamp': datetime.now(),
            'location': f"({lat}, {lon})",
            'source': 'WeatherNext'
        }
        
        # Extract temperature if available
        if 'temperature_2m' in forecast_info:
            temp_k = forecast_info['temperature_2m']
            temp_f = (temp_k - 273.15) * 9/5 + 32  # Kelvin to Fahrenheit
            forecast_data['temperature_f'] = temp_f
            print(f"  Temperature: {temp_f:.1f}°F")
        
        # Extract other variables
        for key, value in forecast_info.items():
            if key not in forecast_data:
                forecast_data[key] = value
        
        return pd.DataFrame([forecast_data])
        
    except Exception as e:
        print(f"✗ Error fetching WeatherNext data: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_weathernext_timeseries(lat=KLGA_LAT, lon=KLGA_LON, hours=168):
    """
    Fetch WeatherNext forecast time series.
    
    Args:
        lat: Latitude
        lon: Longitude
        hours: Number of hours to forecast (default: 168 = 7 days)
    
    Returns:
        DataFrame with hourly forecast
    """
    
    if not initialize_earth_engine():
        return None
    
    print(f"\nFetching WeatherNext time series for ({lat}, {lon})...")
    
    try:
        dataset_id = 'projects/gcp-public-data-weathernext/assets/126478713_1_0'
        point = ee.Geometry.Point([lon, lat])
        
        # Get recent forecasts
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        collection = ee.ImageCollection(dataset_id).filterDate(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        ).filterBounds(point)
        
        # Get collection info
        collection_list = collection.toList(collection.size())
        size = collection_list.size().getInfo()
        
        print(f"✓ Found {size} forecast images")
        
        if size == 0:
            print("✗ No forecast data available")
            return None
        
        # Extract data from each image
        forecast_list = []
        
        for i in range(min(size, 24)):  # Limit to 24 images (hourly for 1 day)
            image = ee.Image(collection_list.get(i))
            
            # Get timestamp
            timestamp = image.get('system:time_start').getInfo()
            forecast_time = datetime.fromtimestamp(timestamp / 1000)
            
            # Sample at point
            values = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=1000
            ).getInfo()
            
            # Parse temperature
            if 'temperature_2m' in values:
                temp_k = values['temperature_2m']
                temp_f = (temp_k - 273.15) * 9/5 + 32
                
                forecast_list.append({
                    'timestamp': forecast_time,
                    'temperature_f': temp_f,
                    'source': 'WeatherNext'
                })
        
        if forecast_list:
            df = pd.DataFrame(forecast_list)
            print(f"✓ Retrieved {len(df)} hourly forecasts")
            return df
        else:
            print("✗ No data extracted")
            return None
            
    except Exception as e:
        print(f"✗ Error fetching time series: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_forecasts():
    """Compare WeatherNext with NWS forecast."""
    
    print("=" * 70)
    print("WeatherNext vs NWS Forecast Comparison")
    print("=" * 70)
    
    # Get WeatherNext
    wn_forecast = get_weathernext_forecast()
    
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
                print(f"\nNWS Forecast High: {nws_high:.1f}°F")
    except Exception as e:
        print(f"Could not fetch NWS: {e}")
    
    if wn_forecast is not None and 'temperature_f' in wn_forecast.columns:
        wn_temp = wn_forecast['temperature_f'].iloc[0]
        print(f"WeatherNext: {wn_temp:.1f}°F")
        
        if 'nws_high' in locals():
            diff = wn_temp - nws_high
            print(f"\nDifference: {diff:+.1f}°F")

if __name__ == "__main__":
    print("=" * 70)
    print("WeatherNext Forecast Fetcher")
    print("=" * 70)
    print()
    print("Note: This requires Google Earth Engine authentication")
    print("Setup: https://developers.google.com/earth-engine/guides/python_install")
    print()
    
    # Try to fetch forecast
    forecast = get_weathernext_forecast()
    
    if forecast is not None:
        print("\n" + "=" * 70)
        print("Forecast Data:")
        print("=" * 70)
        print(forecast.to_string())
        
        # Save to file
        output_file = 'data/raw/weathernext_forecast.csv'
        forecast.to_csv(output_file, index=False)
        print(f"\n✓ Saved to: {output_file}")
    else:
        print("\n" + "=" * 70)
        print("Setup Instructions:")
        print("=" * 70)
        print("1. Install: pip install earthengine-api")
        print("2. Authenticate: earthengine authenticate")
        print("3. Run this script again")
