"""
Test script for NWS Weather Scraper.
Fetches recent weather data from National Weather Service API.

Run: python scripts/nws_test_scraper.py
"""

import sys
from datetime import date, timedelta

sys.path.insert(0, '.')

from src.data.weather_scraper import WeatherScraper, WeatherDataError


def main():
    print("=" * 60)
    print("Testing NWS Weather Scraper (KLGA - LaGuardia Airport)")
    print("=" * 60)
    
    # Initialize scraper
    scraper = WeatherScraper(station_id="KLGA")
    print(f"\nStation ID: {scraper.station_id}")
    print(f"Rate limit delay: {scraper._rate_limit_delay}s")
    
    today = date.today()
    yesterday = today - timedelta(days=1)
    
    # Test 1: Fetch yesterday's data
    print(f"\n--- Test 1: Fetching single day ({yesterday}) ---")
    try:
        result = scraper.fetch_daily_history(yesterday)
        print(f"Date: {result['date']}")
        print(f"Max Temp: {result['max_temp']}°F")
        print(f"Min Temp: {result['min_temp']}°F")
        print(f"Avg Humidity: {result['avg_humidity']}%")
        print(f"Avg Wind Speed: {result['avg_wind_speed']} mph")
        print(f"Total Precipitation: {result['total_precipitation']} in")
    except WeatherDataError as e:
        print(f"Error: {e}")
    
    # Test 2: Fetch last 5 days
    start_date = today - timedelta(days=5)
    end_date = yesterday
    print(f"\n--- Test 2: Fetching date range ({start_date} to {end_date}) ---")
    
    try:
        df = scraper.fetch_date_range(start_date, end_date)
        print("\nDataFrame:")
        print(df)
        
        print("\nSummary:")
        print(f"  Days with data: {df['max_temp'].notna().sum()}")
        print(f"  Avg High: {df['max_temp'].mean():.1f}°F")
        print(f"  Avg Low: {df['min_temp'].mean():.1f}°F")
        
        # Save to CSV
        output_path = "data/raw/nws_klga_recent.csv"
        scraper.save_to_csv(df, output_path)
        
        # Verify load
        loaded_df = scraper.load_from_csv(output_path)
        print(f"\nLoaded back from CSV: {len(loaded_df)} rows")
        
    except WeatherDataError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
