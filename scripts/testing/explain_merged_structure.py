"""
Explain the structure of the merged backtest data
"""
import pandas as pd
import ast

df = pd.read_csv('data/processed/backtest_data_combined.csv')

print('=' * 70)
print('MERGED FILE STRUCTURE EXPLANATION')
print('=' * 70)

print('\n1. COLUMNS IN MERGED FILE:')
print('-' * 70)
for col in df.columns:
    print(f'  - {col}')

print('\n2. SAMPLE RECORD WITH ODDS DATA:')
print('-' * 70)
sample = df[df['threshold'].notna()].iloc[0]
print(f'Date: {sample["date"]}')
print(f'Forecast Date: {sample["forecast_date"]}')
print(f'Forecasted High: {sample["forecasted_high"]:.1f}°F')
print(f'Actual High: {sample["actual_high"]:.1f}°F')
print(f'Actual Low: {sample["actual_low"]:.1f}°F')
print(f'Peak Time: {sample["peak_time"]}')
print(f'\nOdds Data (stored as lists):')
print(f'  Thresholds: {sample["threshold"][:100]}...')
print(f'  Probabilities: {sample["yes_probability"][:100]}...')

print('\n3. SAMPLE RECORD WITHOUT ODDS DATA:')
print('-' * 70)
sample2 = df[df['threshold'].isna()].iloc[0]
print(f'Date: {sample2["date"]}')
print(f'Forecast Date: {sample2["forecast_date"]}')
print(f'Forecasted High: {sample2["forecasted_high"]:.1f}°F')
print(f'Actual High: {sample2["actual_high"]:.1f}°F')
print(f'Odds Data: {sample2["threshold"]} (missing)')

print('\n4. DATA ORGANIZATION:')
print('-' * 70)
print('Each row represents ONE FORECAST for a specific target date.')
print('Multiple forecasts can exist for the same target date (made on different days).')
print('')
print('For odds data:')
print('  - Each event_date can have MULTIPLE thresholds (e.g., 36-37, 38-39, ≤35, ≥46)')
print('  - These are stored as LISTS in the merged file')
print('  - One row per forecast contains ALL thresholds for that date')

print('\n5. TIMING OF ODDS DATA:')
print('-' * 70)
# Check raw odds to show timing
raw_odds = pd.read_csv('data/raw/polymarket_odds_history.csv')
sample_date = '2025-02-01'
sample_odds = raw_odds[raw_odds['event_date'] == sample_date]
print(f'Example: Event date {sample_date}')
print(f'  Fetch timestamp: {sample_odds.iloc[0]["fetch_timestamp"]}')
print(f'  Number of thresholds: {len(sample_odds)}')
print(f'  Thresholds available: {sample_odds["threshold"].tolist()}')

print('\n6. KEY INSIGHTS:')
print('-' * 70)
print('✓ Odds are fetched at a SPECIFIC TIME (fetch_timestamp) for each event_date')
print('✓ The merged file AGGREGATES all thresholds for that fetch into lists')
print('✓ You are NOT missing data - all thresholds are preserved in the lists')
print('✓ For backtesting, you would parse these lists to find relevant thresholds')

print('\n7. RECOMMENDATION FOR NEXT TASKS:')
print('-' * 70)
print('The next tasks should likely:')
print('  1. Parse the list columns back into individual threshold records')
print('  2. Match forecasted_high against the appropriate threshold ranges')
print('  3. Calculate expected value based on probabilities and payoffs')
print('  4. Determine optimal betting strategy')
print('')
print('Alternatively, tasks could work directly with raw files for more flexibility.')

print('\n' + '=' * 70)
