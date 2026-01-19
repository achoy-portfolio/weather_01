import pandas as pd

df = pd.read_csv('data/raw/historical_forecasts.csv')
df['forecast_datetime'] = pd.to_datetime(df['forecast_date'] + ' ' + df['forecast_time'])
df['valid_time'] = pd.to_datetime(df['valid_time'])

print("Checking if forecasts change over time...")
print("="*70)

# Check 20 random target times
sample_targets = df['valid_time'].unique()[::100][:20]

variations_found = 0
for target in sample_targets:
    forecasts = df[df['valid_time'] == target]
    temps_by_day = forecasts.groupby('forecast_date')['temperature'].first()
    
    if len(temps_by_day) > 1:
        std = temps_by_day.std()
        if std > 0.1:
            variations_found += 1
            print(f"\n✓ {target}")
            print(f"  Forecasts vary: {temps_by_day.min():.1f}°F to {temps_by_day.max():.1f}°F (std={std:.2f})")
            print(f"  Issued on {len(temps_by_day)} different days")

if variations_found == 0:
    print("\n❌ NO VARIATION FOUND!")
    print("All forecasts for the same target time are identical,")
    print("regardless of when they were issued.")
    print("\nThis means your forecast data is NOT capturing forecast updates.")
else:
    print(f"\n✓ Found {variations_found} cases where forecasts changed over time")
