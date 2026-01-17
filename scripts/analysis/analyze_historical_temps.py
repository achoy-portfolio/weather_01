"""
Analyze historical temperature data from Open-Meteo for KLGA.
Shows actual temperatures and typical day-to-day variability.
This helps estimate forecast uncertainty.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

KLGA_LAT = 40.7769
KLGA_LON = -73.8740

def get_historical_temps(start_date, end_date, lat=KLGA_LAT, lon=KLGA_LON):
    """Get actual historical temperatures from Open-Meteo."""
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean',
        'temperature_unit': 'fahrenheit',
        'timezone': 'America/New_York'
    }
    
    print(f"Fetching historical temps from {start_date} to {end_date}...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame({
            'date': pd.to_datetime(data['daily']['time']),
            'temp_max': data['daily']['temperature_2m_max'],
            'temp_min': data['daily']['temperature_2m_min'],
            'temp_mean': data['daily']['temperature_2m_mean']
        })
        
        print(f"✓ Retrieved {len(df)} days")
        return df
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def analyze_forecast_difficulty(df):
    """
    Analyze how predictable temperatures are based on historical patterns.
    This simulates forecast accuracy by using persistence and climatology.
    """
    
    print("\n" + "=" * 70)
    print("FORECAST DIFFICULTY ANALYSIS")
    print("=" * 70)
    
    # Calculate day-to-day changes
    df['temp_change'] = df['temp_max'].diff()
    
    # Persistence forecast: tomorrow = today
    df['persistence_forecast'] = df['temp_max'].shift(1)
    df['persistence_error'] = df['temp_max'] - df['persistence_forecast']
    
    # Calculate statistics
    print(f"\nTemperature Statistics:")
    print(f"  Mean high: {df['temp_max'].mean():.1f}°F")
    print(f"  Std dev: {df['temp_max'].std():.1f}°F")
    print(f"  Range: {df['temp_max'].min():.1f}°F to {df['temp_max'].max():.1f}°F")
    
    print(f"\nDay-to-Day Variability:")
    print(f"  Mean change: {df['temp_change'].mean():.1f}°F")
    print(f"  Std dev of changes: {df['temp_change'].std():.1f}°F")
    print(f"  Max increase: {df['temp_change'].max():.1f}°F")
    print(f"  Max decrease: {df['temp_change'].min():.1f}°F")
    
    print(f"\nPersistence Forecast (tomorrow = today):")
    print(f"  MAE: {df['persistence_error'].abs().mean():.1f}°F")
    print(f"  RMSE: {np.sqrt((df['persistence_error']**2).mean()):.1f}°F")
    print(f"  Bias: {df['persistence_error'].mean():+.1f}°F")
    
    print(f"\n💡 Interpretation:")
    mae = df['persistence_error'].abs().mean()
    if mae < 4:
        print(f"  Low variability - temperatures are predictable")
        print(f"  Good forecasts should have MAE < 3°F")
    elif mae < 6:
        print(f"  Moderate variability - typical winter conditions")
        print(f"  Good forecasts should have MAE < 4°F")
    else:
        print(f"  High variability - temperatures change rapidly")
        print(f"  Even good forecasts may have MAE > 5°F")
    
    return df

def compare_december_2025():
    """Analyze December 2025 temperatures at KLGA."""
    
    print("=" * 70)
    print("DECEMBER 2025 TEMPERATURE ANALYSIS - KLGA")
    print("=" * 70)
    print()
    
    # Get December 2025 data
    df = get_historical_temps('2025-12-01', '2025-12-31')
    
    if df is None:
        return
    
    # Analyze
    df = analyze_forecast_difficulty(df)
    
    # Show daily temps
    print("\n" + "=" * 70)
    print("DAILY TEMPERATURES")
    print("=" * 70)
    print(f"{'Date':<12} {'High':<8} {'Low':<8} {'Change':<10}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        high = row['temp_max']
        low = row['temp_min']
        change = row['temp_change']
        
        if pd.notna(change):
            print(f"{date_str:<12} {high:>6.1f}°F {low:>6.1f}°F {change:>+7.1f}°F")
        else:
            print(f"{date_str:<12} {high:>6.1f}°F {low:>6.1f}°F {'N/A':>10}")
    
    # Save results
    df.to_csv('data/raw/december_2025_temps.csv', index=False)
    print(f"\n✓ Saved to: data/raw/december_2025_temps.csv")
    
    # Plot
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['temp_max'], 'r-', label='High', linewidth=2)
        plt.plot(df['date'], df['temp_min'], 'b-', label='Low', linewidth=2)
        plt.fill_between(df['date'], df['temp_min'], df['temp_max'], alpha=0.3)
        plt.xlabel('Date')
        plt.ylabel('Temperature (°F)')
        plt.title('KLGA December 2025 Temperatures')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/raw/december_2025_temps.png', dpi=150)
        print(f"✓ Chart saved to: data/raw/december_2025_temps.png")
    except Exception as e:
        print(f"Could not create chart: {e}")
    
    return df

if __name__ == "__main__":
    df = compare_december_2025()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nThis shows actual December 2025 temperatures.")
    print("Persistence forecast MAE represents baseline difficulty.")
    print("Professional forecasts (NWS, Open-Meteo) should beat this by 1-2°F.")
    print("\nTo track actual forecast accuracy, run:")
    print("  python scripts/track_forecast_accuracy.py")
    print("daily to save forecasts and verify them later.")
