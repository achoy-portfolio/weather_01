import pandas as pd
import matplotlib.pyplot as plt

# Load actual temperatures for Feb 1
actuals = pd.read_csv('data/raw/actual_temperatures.csv')
actuals['timestamp'] = pd.to_datetime(actuals['timestamp'])
feb1_actual = actuals[actuals['timestamp'].dt.date == pd.to_datetime('2025-02-01').date()].copy()
feb1_actual['hour'] = feb1_actual['timestamp'].dt.hour

# Load forecasts issued on Feb 1 for Feb 1
forecasts = pd.read_csv('data/raw/historical_forecasts.csv')
forecasts['forecast_datetime'] = pd.to_datetime(forecasts['forecast_date'] + ' ' + forecasts['forecast_time'])
forecasts['valid_time'] = pd.to_datetime(forecasts['valid_time'])

# Get forecast issued at midnight on Feb 1
feb1_forecast = forecasts[
    (forecasts['forecast_date'] == '2025-02-01') & 
    (forecasts['forecast_time'] == '00:00') &
    (forecasts['valid_time'].dt.date == pd.to_datetime('2025-02-01').date())
].copy()
feb1_forecast['hour'] = feb1_forecast['valid_time'].dt.hour

print("="*70)
print("REAL EXAMPLE: February 1st, 2025")
print("="*70)

print("\nForecast issued at: 12:00 AM (midnight) on Feb 1st")
print("Target: Predict the daily high for Feb 1st")

print("\n" + "-"*70)
print("Hour | Actual Temp | Forecasted Temp | Difference")
print("-"*70)

for hour in range(24):
    actual_row = feb1_actual[feb1_actual['hour'] == hour]
    forecast_row = feb1_forecast[feb1_forecast['hour'] == hour]
    
    if not actual_row.empty and not forecast_row.empty:
        actual_temp = actual_row['temperature_f'].values[0]
        forecast_temp = forecast_row['temperature'].values[0]
        diff = forecast_temp - actual_temp
        
        marker = " ← ACTUAL HIGH" if actual_temp == feb1_actual['temperature_f'].max() else ""
        marker += " ← FORECAST HIGH" if forecast_temp == feb1_forecast['temperature'].max() else ""
        
        print(f"{hour:2d}   | {actual_temp:6.1f}°F    | {forecast_temp:6.1f}°F         | {diff:+6.1f}°F{marker}")

print("-"*70)

actual_high = feb1_actual['temperature_f'].max()
forecast_high = feb1_forecast['temperature'].max()
error = forecast_high - actual_high

print(f"\nACTUAL DAILY HIGH:     {actual_high:.1f}°F (occurred at {feb1_actual[feb1_actual['temperature_f'] == actual_high]['hour'].values[0]}:00)")
print(f"FORECASTED DAILY HIGH: {forecast_high:.1f}°F")
print(f"ERROR:                 {error:+.1f}°F")

print("\n" + "="*70)
print("THE PROBLEM:")
print("="*70)
print(f"\nThe forecast at midnight predicted temps would reach {forecast_high:.1f}°F")
print(f"But the actual high was only {actual_high:.1f}°F (at 1 AM)")
print(f"\nThe forecast was predicting FUTURE hours would be warmer,")
print(f"but it couldn't predict what ALREADY happened (the 1 AM peak).")
print(f"\nThis is why 0-day forecasts have {error:.1f}°F error for this day,")
print(f"even though hourly forecasts are generally accurate (~2°F).")

# Create a simple plot
plt.figure(figsize=(12, 6))
plt.plot(feb1_actual['hour'], feb1_actual['temperature_f'], 'b-o', label='Actual', linewidth=2)
plt.plot(feb1_forecast['hour'], feb1_forecast['temperature'], 'r--s', label='Forecast (issued at midnight)', linewidth=2)
plt.axhline(y=actual_high, color='b', linestyle=':', alpha=0.5, label=f'Actual High: {actual_high:.1f}°F')
plt.axhline(y=forecast_high, color='r', linestyle=':', alpha=0.5, label=f'Forecast High: {forecast_high:.1f}°F')
plt.xlabel('Hour of Day')
plt.ylabel('Temperature (°F)')
plt.title('February 1st, 2025: 0-Day Forecast vs Actual\n(Forecast issued at midnight on Feb 1st)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig('feb1_0day_forecast_example.png', dpi=150)
print(f"\nChart saved to: feb1_0day_forecast_example.png")
