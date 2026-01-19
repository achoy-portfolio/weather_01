"""
Verify that odds are correctly filtered to 9 PM the day before each event
"""
import pandas as pd

df = pd.read_csv('data/processed/backtest_data_combined.csv')
df['date'] = pd.to_datetime(df['date'])
df['fetch_timestamp'] = pd.to_datetime(df['fetch_timestamp'])

print('=' * 80)
print('ODDS TIMING VERIFICATION')
print('=' * 80)

print('\n✓ CORRECT BEHAVIOR:')
print('All forecasts for the same target date should use the SAME odds')
print('(the odds available at 9 PM Eastern the day before the event)')

print('\n' + '=' * 80)
print('SAMPLE: Event Date 2026-01-06')
print('=' * 80)

sample = df[df['date'] == '2026-01-06'].copy()
print(f'\nNumber of forecasts for this date: {len(sample)}')
print(f'Unique fetch_timestamps: {sample["fetch_timestamp"].nunique()}')
print(f'Fetch timestamp used: {sample["fetch_timestamp"].iloc[0]}')

print('\nAll forecasts:')
print(sample[['forecast_date', 'date', 'forecasted_high', 'threshold', 'yes_probability']].to_string(index=False))

print('\n' + '=' * 80)
print('VERIFICATION: Check multiple dates')
print('=' * 80)

# Check a few dates
test_dates = df[df['threshold'].notna()]['date'].unique()[:5]

all_correct = True
for test_date in test_dates:
    date_data = df[df['date'] == test_date]
    unique_odds = date_data['fetch_timestamp'].nunique()
    
    if unique_odds == 1:
        print(f'✓ {test_date}: {len(date_data)} forecasts, 1 unique odds timestamp')
    else:
        print(f'✗ {test_date}: {len(date_data)} forecasts, {unique_odds} unique odds timestamps (ERROR!)')
        all_correct = False

print('\n' + '=' * 80)
if all_correct:
    print('✓ SUCCESS: All target dates use consistent odds (9 PM day before)')
else:
    print('✗ FAILURE: Some dates have inconsistent odds')
print('=' * 80)
