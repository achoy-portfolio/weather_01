"""
Track forecast accuracy by saving daily forecasts and comparing to actuals.
Run this daily to build a performance database.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import json

KLGA_LAT = 40.7769
KLGA_LON = -73.8740

def get_actual_temperature(date, lat=KLGA_LAT, lon=KLGA_LON):
    """
    Get actual observed temperature from Open-Meteo historical API.
    
    Args:
        date: Date string 'YYYY-MM-DD' or datetime object
    
    Returns:
        dict with actual high/low temps
    """
    if isinstance(date, datetime):
        date = date.strftime('%Y-%m-%d')
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': date,
        'end_date': date,
        'daily': 'temperature_2m_max,temperature_2m_min',
        'temperature_unit': 'fahrenheit',
        'timezone': 'America/New_York'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['daily']['time']:
            high = data['daily']['temperature_2m_max'][0]
            low = data['daily']['temperature_2m_min'][0]
            return {'high': high, 'low': low, 'date': date}
    except Exception as e:
        print(f"Error fetching actual temp: {e}")
    
    return None

def save_todays_forecast():
    """Save today's forecast for tomorrow to track accuracy later."""
    
    from compare_all_forecasts import compare_all_forecasts
    
    print("Saving today's forecast for tomorrow...")
    
    result = compare_all_forecasts()
    
    if result is None:
        print("✗ Could not get forecasts")
        return
    
    # Create forecast record
    forecast_record = {
        'forecast_date': datetime.now().strftime('%Y-%m-%d'),
        'target_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
        'consensus_high': float(result['consensus_high']),
        'consensus_low': float(result['consensus_low']),
        'uncertainty_high': float(result['uncertainty_high']),
        'uncertainty_low': float(result['uncertainty_low']),
        'num_sources': int(result['num_sources']),
        'forecasts': [
            {
                'source': fc['source'],
                'high': float(fc['high']),
                'low': float(fc['low'])
            }
            for fc in result['forecasts']
        ]
    }
    
    # Save to file
    forecast_file = 'data/forecasts/forecast_history.jsonl'
    os.makedirs('data/forecasts', exist_ok=True)
    
    with open(forecast_file, 'a') as f:
        f.write(json.dumps(forecast_record) + '\n')
    
    print(f"✓ Saved forecast: {forecast_record['target_date']} - High: {forecast_record['consensus_high']:.1f}°F")
    
    return forecast_record

def verify_past_forecasts():
    """Check accuracy of past forecasts by comparing to actual temps."""
    
    forecast_file = 'data/forecasts/forecast_history.jsonl'
    
    if not os.path.exists(forecast_file):
        print("No forecast history found. Run save_todays_forecast() first.")
        return None
    
    print("Verifying past forecasts...")
    print("=" * 70)
    
    # Load all forecasts
    forecasts = []
    with open(forecast_file, 'r') as f:
        for line in f:
            forecasts.append(json.loads(line))
    
    # Check each forecast
    results = []
    
    for fc in forecasts:
        target_date = fc['target_date']
        
        # Only verify if target date is in the past
        if datetime.strptime(target_date, '%Y-%m-%d').date() >= datetime.now().date():
            continue
        
        # Get actual temperature
        actual = get_actual_temperature(target_date)
        
        if actual:
            error_high = fc['consensus_high'] - actual['high']
            error_low = fc['consensus_low'] - actual['low']
            
            results.append({
                'target_date': target_date,
                'forecast_high': fc['consensus_high'],
                'actual_high': actual['high'],
                'error_high': error_high,
                'forecast_low': fc['consensus_low'],
                'actual_low': actual['low'],
                'error_low': error_low,
                'uncertainty': fc['uncertainty_high'],
                'num_sources': fc['num_sources']
            })
            
            print(f"{target_date}: Forecast {fc['consensus_high']:.1f}°F, Actual {actual['high']:.1f}°F, Error {error_high:+.1f}°F")
    
    if not results:
        print("No past forecasts to verify yet.")
        return None
    
    # Calculate statistics
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("FORECAST ACCURACY STATISTICS")
    print("=" * 70)
    print(f"Forecasts verified: {len(df)}")
    print(f"\nHigh Temperature:")
    print(f"  Mean Error: {df['error_high'].mean():+.2f}°F")
    print(f"  MAE: {df['error_high'].abs().mean():.2f}°F")
    print(f"  RMSE: {(df['error_high']**2).mean()**0.5:.2f}°F")
    print(f"  Std Dev: {df['error_high'].std():.2f}°F")
    
    print(f"\nLow Temperature:")
    print(f"  Mean Error: {df['error_low'].mean():+.2f}°F")
    print(f"  MAE: {df['error_low'].abs().mean():.2f}°F")
    print(f"  RMSE: {(df['error_low']**2).mean()**0.5:.2f}°F")
    
    # Save results
    df.to_csv('data/forecasts/forecast_accuracy.csv', index=False)
    print(f"\n✓ Saved to: data/forecasts/forecast_accuracy.csv")
    
    return df

if __name__ == "__main__":
    print("=" * 70)
    print("FORECAST ACCURACY TRACKER")
    print("=" * 70)
    print()
    
    # Save today's forecast
    print("Step 1: Save today's forecast")
    print("-" * 70)
    save_todays_forecast()
    
    print("\n" + "=" * 70)
    print("Step 2: Verify past forecasts")
    print("-" * 70)
    verify_past_forecasts()
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print("\nRun this script daily to build forecast accuracy history.")
    print("After a few days, you'll have statistics on forecast performance.")
