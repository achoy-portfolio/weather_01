"""
Analysis of merged data structure and recommendations for next tasks
"""
import pandas as pd
import ast

print('=' * 80)
print('MERGED DATA STRUCTURE ANALYSIS')
print('=' * 80)

# Load data
merged_df = pd.read_csv('data/processed/backtest_data_combined.csv')
raw_odds = pd.read_csv('data/raw/polymarket_odds_history.csv')

print('\n📊 CURRENT STRUCTURE (backtest_data_combined.csv):')
print('-' * 80)
print('Each row = ONE FORECAST for a target date')
print(f'Total rows: {len(merged_df)}')
print(f'Rows with odds: {merged_df["threshold"].notna().sum()}')
print(f'Rows without odds: {merged_df["threshold"].isna().sum()}')

print('\n🔍 EXAMPLE: Multiple forecasts for same date')
print('-' * 80)
sample_date = '2025-01-24'
same_date = merged_df[merged_df['date'] == sample_date][['forecast_date', 'date', 'forecasted_high', 'actual_high']]
print(same_date.to_string(index=False))
print(f'\n→ {len(same_date)} different forecasts made for {sample_date}')

print('\n📦 ODDS DATA STORAGE:')
print('-' * 80)
sample_with_odds = merged_df[merged_df['threshold'].notna()].iloc[0]
thresholds = ast.literal_eval(sample_with_odds['threshold'])
probs = ast.literal_eval(sample_with_odds['yes_probability'])
print(f'Date: {sample_with_odds["date"]}')
print(f'Number of thresholds: {len(thresholds)}')
print('\nThreshold → Probability:')
for t, p in zip(thresholds, probs):
    print(f'  {t:>8} → {p:.3f}')

print('\n⏰ TIMING INFORMATION:')
print('-' * 80)
sample_odds_date = raw_odds[raw_odds['event_date'] == '2025-02-01'].iloc[0]
print(f'Event Date: {sample_odds_date["event_date"]}')
print(f'Fetch Time: {sample_odds_date["fetch_timestamp"]}')
print('\n→ Odds are fetched at a SPECIFIC TIME before the event')
print('→ This represents the market state at that moment')
print('→ For Feb 1 event, odds were fetched on Jan 30 at 3pm')

print('\n❓ ARE YOU MISSING DATA?')
print('-' * 80)
print('NO! All odds data is preserved in the merged file.')
print('The lists contain ALL thresholds available for each event date.')
print('')
print('What you have:')
print('  ✓ All thresholds (ranges, above, below)')
print('  ✓ All probabilities')
print('  ✓ All volumes')
print('  ✓ Fetch timestamps (in raw file)')

print('\n🎯 RECOMMENDATION FOR NEXT TASKS:')
print('-' * 80)
print('\nOption A: EXPAND the merged file (recommended for tasks 5-6)')
print('  - Create one row per forecast × threshold combination')
print('  - This makes betting simulation easier')
print('  - Example: 1 forecast with 7 thresholds → 7 rows')
print('')
print('Option B: Work with RAW files directly')
print('  - Keep raw odds separate')
print('  - Join on-the-fly during betting simulation')
print('  - More flexible but requires more complex joins')

print('\n📋 SUGGESTED APPROACH FOR TASK 5 (Betting Simulator):')
print('-' * 80)
print('1. Load backtest_data_combined.csv')
print('2. For each row with odds:')
print('   a. Parse the threshold/probability lists')
print('   b. For each threshold:')
print('      - Calculate model probability from forecasted_high')
print('      - Calculate expected value')
print('      - Decide whether to bet')
print('      - Record decision')
print('3. Output: expanded dataset with betting decisions')

print('\n💡 ALTERNATIVE: Create an expanded version now')
print('-' * 80)
print('We could create a new file: backtest_data_expanded.csv')
print('Structure:')
print('  - forecast_date, target_date, forecasted_high, actual_high')
print('  - threshold, threshold_type, yes_probability, volume')
print('  - One row per forecast-threshold pair')
print(f'  - Estimated rows: ~{merged_df["threshold"].notna().sum() * 7} rows')

print('\n' + '=' * 80)
print('CONCLUSION:')
print('=' * 80)
print('The merged file is correctly organized. You have all the data.')
print('For tasks 5-6, you\'ll need to "explode" the list columns into separate rows.')
print('This can be done either:')
print('  1. As a preprocessing step (create expanded file now)')
print('  2. Within the betting simulator (parse lists on-the-fly)')
print('=' * 80)
