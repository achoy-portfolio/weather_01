"""Verify the odds data has multiple timestamps"""
import pandas as pd

df = pd.read_csv('data/raw/polymarket_odds_history.csv')
df['fetch_timestamp'] = pd.to_datetime(df['fetch_timestamp'])

sample = df[df['event_date'] == '2026-01-03'].sort_values('fetch_timestamp')

print('Odds for 2026-01-03:')
print(f'Total records: {len(sample)}')
print(f'Unique timestamps: {sample["fetch_timestamp"].nunique()}')
print(f'Unique thresholds: {sample["threshold"].nunique()}')
print(f'\nTime range: {sample["fetch_timestamp"].min()} to {sample["fetch_timestamp"].max()}')

print(f'\nSample - 28-29 threshold over time (showing how odds changed):')
threshold_sample = sample[sample['threshold'] == '28-29'][['fetch_timestamp', 'yes_probability']].head(10)
print(threshold_sample.to_string(index=False))

print(f'\nAll thresholds at one timestamp (2026-01-01 07:00):')
time_sample = sample[sample['fetch_timestamp'].dt.hour == 7].groupby('threshold').first()[['yes_probability']]
print(time_sample)
