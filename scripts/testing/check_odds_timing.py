"""
Check the timing of odds data to understand the issue
"""
import pandas as pd

df = pd.read_csv('data/raw/polymarket_odds_history.csv')
df['fetch_timestamp'] = pd.to_datetime(df['fetch_timestamp'])
df['event_date'] = pd.to_datetime(df['event_date'])

print('=' * 80)
print('ODDS TIMING ANALYSIS')
print('=' * 80)

# Check a specific event date
sample_date = '2026-01-06'
sample = df[df['event_date'] == sample_date].sort_values('fetch_timestamp')

print(f'\nOdds available for event date: {sample_date}')
print('-' * 80)

# Group by fetch_timestamp to see unique fetches
unique_fetches = sample.groupby('fetch_timestamp').agg({
    'threshold': lambda x: list(x),
    'yes_probability': lambda x: list(x)
}).reset_index()

print(f'Number of different fetch times: {len(unique_fetches)}')
print('\nFetch timestamps:')
for _, row in unique_fetches.iterrows():
    print(f'  {row["fetch_timestamp"]} - {len(row["threshold"])} thresholds')

print('\n' + '=' * 80)
print('THE PROBLEM:')
print('=' * 80)
print('The current merger joins odds by event_date only.')
print('It does NOT consider the forecast_date or fetch_timestamp.')
print('This means ALL forecasts for the same target date get the SAME odds.')
print('')
print('What we need:')
print('  - Match odds based on WHEN the forecast was made')
print('  - Use odds from 9pm the day before resolution')
print('  - Or use the most recent odds available before the forecast time')

print('\n' + '=' * 80)
print('SOLUTION:')
print('=' * 80)
print('We need to modify the merger to:')
print('1. Filter odds to a specific time (e.g., 9pm day before)')
print('2. OR match odds based on forecast_date (use odds available at forecast time)')
print('3. This requires joining on both event_date AND timestamp logic')
