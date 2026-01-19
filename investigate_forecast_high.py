import pandas as pd
import numpy as np

# Load historical forecasts
hist_df = pd.read_csv('data/raw/historical_forecasts.csv')

# Look at one specific day to understand the data structure
print("="*70)
print("EXAMPLE: 2025-01-21 FORECASTS")
print("="*70)

jan21 = hist_df[hist_df['forecast_date'] == '2025-01-21'].copy()
print(f"\nTotal rows for 2025-01-21: {len(jan21)}")
print(f"Unique forecast_time values: {sorted(jan21['forecast_time'].unique())}")

# Show forecasts issued at 00:00 (midnight)
print("\n" + "="*70)
print("FORECASTS ISSUED AT 00:00 (MIDNIGHT) ON 2025-01-21")
print("="*70)
midnight = jan21[jan21['forecast_time'] == '00:00']
print(f"Number of hourly forecasts: {len(midnight)}")
print(f"Valid times range: {midnight['valid_time'].min()} to {midnight['valid_time'].max()}")
print(f"Temperature range: {midnight['temperature'].min():.1f}°F to {midnight['temperature'].max():.1f}°F")
print(f"\nMax temperature in this forecast: {midnight['temperature'].max():.1f}°F")

# Check what the actual high was
actual_df = pd.read_csv('data/raw/actual_temperatures.csv')
actual_jan21 = actual_df[actual_df['timestamp'].str.startswith('2025-01-21')]
print(f"\nActual high on 2025-01-21: {actual_jan21['temperature_f'].max():.1f}°F")

# Now check the combined data
combined_df = pd.read_csv('data/processed/backtest_data_combined.csv')
combined_jan21 = combined_df[(combined_df['forecast_date'] == '2025-01-21') & 
                              (combined_df['date'] == '2025-01-21')]
print(f"\nIn combined data:")
print(f"  Forecasted high: {combined_jan21['forecasted_high'].values[0]:.1f}°F")
print(f"  Actual high: {combined_jan21['actual_high'].values[0]:.1f}°F")
print(f"  Error: {combined_jan21['forecasted_high'].values[0] - combined_jan21['actual_high'].values[0]:.1f}°F")

print("\n" + "="*70)
print("THE PROBLEM:")
print("="*70)
print("The 0-day forecasts are issued at MIDNIGHT (00:00) on the same day.")
print("At midnight, the forecast doesn't have information about the full day yet.")
print("The forecast at 00:00 is predicting temperatures for the next 24-72 hours,")
print("but the 'forecasted_high' is being extracted from the wrong time window.")
print("\nThe midnight forecast shows a max of 19.5°F (at hour 15 = 3 PM),")
print("but the actual high was 17.2°F. This is actually pretty close!")
print("\nHowever, if we're extracting the wrong value (like the max from the")
print("NEXT day's forecast window), that would explain the large error.")
