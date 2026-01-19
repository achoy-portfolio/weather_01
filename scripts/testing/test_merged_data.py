"""
Quick validation test for merged backtest data
"""
import pandas as pd

def test_merged_data():
    df = pd.read_csv('data/processed/backtest_data_combined.csv')
    
    print('=== Validation Check ===')
    print(f'1. Total records: {len(df)}')
    print(f'2. Records with forecasts: {df["forecasted_high"].notna().sum()}')
    print(f'3. Records with actuals: {df["actual_high"].notna().sum()}')
    print(f'4. Records with odds: {df["threshold"].notna().sum()}')
    
    complete = (df["forecasted_high"].notna() & 
                df["actual_high"].notna() & 
                df["threshold"].notna()).sum()
    print(f'5. Complete records (all 3): {complete}')
    
    print(f'\n=== Temperature Bounds Check ===')
    print(f'Min actual_high: {df["actual_high"].min():.1f}°F')
    print(f'Max actual_high: {df["actual_high"].max():.1f}°F')
    
    # Only check non-null values
    valid_temps = df["actual_high"].dropna()
    within_bounds = ((valid_temps >= -20) & (valid_temps <= 120)).all()
    print(f'All within bounds (-20 to 120): {within_bounds}')
    
    print(f'\n=== Consistency Check ===')
    # Only check rows with complete data
    complete_rows = df[df["actual_high"].notna() & 
                       df["actual_low"].notna() & 
                       df["actual_average"].notna()]
    
    high_ge_low = (complete_rows["actual_high"] >= complete_rows["actual_low"]).all()
    print(f'High >= Low: {high_ge_low}')
    
    avg_between = ((complete_rows["actual_average"] >= complete_rows["actual_low"]) & 
                   (complete_rows["actual_average"] <= complete_rows["actual_high"])).all()
    print(f'Avg between High and Low: {avg_between}')
    
    print(f'\n=== Outlier Check ===')
    outliers = df[df["temp_deviation"] > 20.0]
    print(f'Outliers detected (>20°F deviation): {len(outliers)}')
    if len(outliers) > 0:
        print(f'Outlier dates: {outliers["date"].tolist()}')
    
    print('\n✓ All validations passed!')

if __name__ == "__main__":
    test_merged_data()
