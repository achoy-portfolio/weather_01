"""Quick test to fetch 5 days of data"""
import sys
sys.path.insert(0, 'scripts/fetching')

from fetch_wunderground_actual import update_temperatures
from datetime import datetime, timedelta

# Fetch just 5 days
start = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

print(f"Fetching data from {start} to {end}")

hourly_df, daily_max_df = update_temperatures(
    start_date=start,
    end_date=end,
    hourly_csv_path='data/raw/wunderground_hourly_temps_test.csv',
    daily_csv_path='data/raw/wunderground_daily_max_temps_test.csv'
)

print("\n" + "=" * 70)
print("HOURLY DATA")
print("=" * 70)
print(hourly_df)
print(f"\nSaved to: data/raw/wunderground_hourly_temps_test.csv")

print("\n" + "=" * 70)
print("DAILY MAX DATA")
print("=" * 70)
print(daily_max_df)
print(f"\nSaved to: data/raw/wunderground_daily_max_temps_test.csv")
