import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/processed/backtest_data_combined.csv')
df['forecast_date'] = pd.to_datetime(df['forecast_date'])
df['date'] = pd.to_datetime(df['date'])
df['lead_time'] = (df['date'] - df['forecast_date']).dt.days
df['error'] = df['forecasted_high'] - df['actual_high']
df['abs_error'] = abs(df['error'])

print("="*70)
print("ERROR ANALYSIS BY LEAD TIME")
print("="*70)

for lt in sorted(df['lead_time'].unique()):
    data = df[df['lead_time'] == lt]
    print(f'\n{lt}-day lead time:')
    print(f'  Count: {len(data)}')
    print(f'  MAE: {data["abs_error"].mean():.2f}°F')
    print(f'  Mean Error (bias): {data["error"].mean():.2f}°F')
    print(f'  Min Error: {data["error"].min():.2f}°F')
    print(f'  Max Error: {data["error"].max():.2f}°F')
    print(f'  Std Dev: {data["error"].std():.2f}°F')

print("\n" + "="*70)
print("SAMPLE 0-DAY FORECASTS (showing source and forecast details)")
print("="*70)

zero_day = df[df['lead_time'] == 0].copy()
print(zero_day[['forecast_date', 'date', 'source', 'forecasted_high', 'actual_high', 'error']].head(20).to_string())

print("\n" + "="*70)
print("CHECKING IF 0-DAY FORECASTS ARE ACTUALLY SAME-DAY")
print("="*70)

# Check if forecast_date == date for 0-day forecasts
zero_day['same_day'] = zero_day['forecast_date'] == zero_day['date']
print(f"All 0-day forecasts are same-day: {zero_day['same_day'].all()}")
print(f"Number of 0-day forecasts: {len(zero_day)}")

# Look at the raw historical_forecasts.csv to understand the structure
print("\n" + "="*70)
print("CHECKING HISTORICAL FORECASTS DATA STRUCTURE")
print("="*70)

hist_df = pd.read_csv('data/raw/historical_forecasts.csv')
print(f"Columns in historical_forecasts.csv: {hist_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(hist_df.head(10))

# Check unique forecast_date and forecast_time combinations
print(f"\nUnique forecast_date values: {hist_df['forecast_date'].nunique()}")
print(f"Unique forecast_time values: {hist_df['forecast_time'].unique()[:10]}")
