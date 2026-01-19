import pandas as pd

# Load historical forecasts
hist_df = pd.read_csv('data/raw/historical_forecasts.csv')
hist_df['forecast_date'] = pd.to_datetime(hist_df['forecast_date'])
hist_df['valid_time'] = pd.to_datetime(hist_df['valid_time'])
hist_df['target_date'] = hist_df['valid_time'].dt.date

print("="*70)
print("ANALYZING HOW FORECASTED_HIGH IS EXTRACTED")
print("="*70)

# Look at 2025-02-01 which had a large error
print("\nExample: Forecast issued on 2025-02-01 for 2025-02-01 (0-day)")
print("-"*70)

feb1_forecast = hist_df[(hist_df['forecast_date'] == '2025-02-01') & 
                         (hist_df['target_date'] == pd.to_datetime('2025-02-01').date())]

print(f"Number of hourly forecasts: {len(feb1_forecast)}")
print(f"Valid times: {feb1_forecast['valid_time'].min()} to {feb1_forecast['valid_time'].max()}")
print(f"Temperature range: {feb1_forecast['temperature'].min():.1f}°F to {feb1_forecast['temperature'].max():.1f}°F")
print(f"\nMAX temperature (what gets used as forecasted_high): {feb1_forecast['temperature'].max():.1f}°F")

# Show the hourly breakdown
print("\nHourly forecast breakdown:")
print(feb1_forecast[['valid_time', 'temperature']].to_string(index=False))

# Check actual
actual_df = pd.read_csv('data/raw/actual_temperatures.csv')
actual_feb1 = actual_df[actual_df['timestamp'].str.startswith('2025-02-01')]
print(f"\nActual high on 2025-02-01: {actual_feb1['temperature_f'].max():.1f}°F")

# Check combined data
combined_df = pd.read_csv('data/processed/backtest_data_combined.csv')
combined_feb1 = combined_df[(combined_df['forecast_date'] == '2025-02-01') & 
                             (combined_df['date'] == '2025-02-01')]
if len(combined_feb1) > 0:
    print(f"\nIn combined data:")
    print(f"  Forecasted high: {combined_feb1['forecasted_high'].values[0]:.1f}°F")
    print(f"  Actual high: {combined_feb1['actual_high'].values[0]:.1f}°F")
    print(f"  Error: {combined_feb1['forecasted_high'].values[0] - combined_feb1['actual_high'].values[0]:.1f}°F")

print("\n" + "="*70)
print("THE ISSUE:")
print("="*70)
print("When a forecast is issued at midnight (00:00) on 2025-02-01,")
print("it contains hourly forecasts for the NEXT 72 hours.")
print("\nThe merge script groups by (forecast_date, target_date) and takes MAX.")
print("So for forecast_date=2025-02-01, target_date=2025-02-01,")
print("it's taking the MAX of all hours that fall on 2025-02-01.")
print("\nBUT: A forecast issued at midnight on 2025-02-01 only has")
print("predictions for hours 0-23 of that day (the remaining hours).")
print("It doesn't have the full day's forecast from the previous evening.")
print("\nThis is why 0-day forecasts have such high error!")
print("They're incomplete forecasts issued partway through the day.")
