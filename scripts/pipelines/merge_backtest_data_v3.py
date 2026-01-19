"""
Enhanced Backtest Data Merger (v3)

Key improvement: For 0-day forecasts, only compare remaining hours.
- If forecast issued at 2 PM, compare forecasted max (2 PM onwards) vs actual max (2 PM onwards)
- This gives a fair comparison of forecast accuracy for same-day predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_historical_forecasts_enhanced(filepath='data/raw/historical_forecasts.csv'):
    """
    Load and aggregate historical forecasts to daily peak forecasts.
    For 0-day forecasts, only use hours AFTER the forecast was issued.
    
    Returns DataFrame with columns: forecast_date, forecast_time, target_date, forecasted_high
    """
    logger.info(f"Loading historical forecasts from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        # Parse dates and times
        df['forecast_datetime'] = pd.to_datetime(df['forecast_date'] + ' ' + df['forecast_time'])
        df['valid_time'] = pd.to_datetime(df['valid_time'])
        
        # Extract target date from valid_time
        df['target_date'] = df['valid_time'].dt.date
        df['forecast_date_only'] = df['forecast_datetime'].dt.date
        
        # For 0-day forecasts, only keep hours AFTER the forecast was issued
        # For 1+ day forecasts, keep all hours
        df['is_same_day'] = df['forecast_date_only'] == df['target_date']
        
        # Filter: for same-day forecasts, only keep future hours
        df_filtered = df[
            (~df['is_same_day']) |  # Keep all non-same-day forecasts
            (df['valid_time'] >= df['forecast_datetime'])  # For same-day, only future hours
        ].copy()
        
        logger.info(f"Filtered {len(df)} rows to {len(df_filtered)} rows (removed past hours from same-day forecasts)")
        
        # Group by forecast_datetime and target_date, get max temperature (peak forecast)
        daily_forecasts = df_filtered.groupby(['forecast_datetime', 'target_date']).agg({
            'temperature': 'max',
            'source': 'first'
        }).reset_index()
        
        daily_forecasts.rename(columns={'temperature': 'forecasted_high'}, inplace=True)
        daily_forecasts['target_date'] = pd.to_datetime(daily_forecasts['target_date'])
        daily_forecasts['forecast_date'] = daily_forecasts['forecast_datetime'].dt.date
        daily_forecasts['forecast_date'] = pd.to_datetime(daily_forecasts['forecast_date'])
        
        logger.info(f"Loaded {len(daily_forecasts)} daily forecasts")
        return daily_forecasts
        
    except FileNotFoundError:
        logger.error(f"Forecast file not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading forecasts: {e}")
        return pd.DataFrame()


def load_actual_temperatures_enhanced(filepath='data/raw/actual_temperatures.csv'):
    """
    Load actual temperatures.
    For 0-day comparison, we'll need to filter by time later.
    
    Returns DataFrame with hourly data and daily aggregates.
    """
    logger.info(f"Loading actual temperatures from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['date'] = pd.to_datetime(df['date'])
        
        # Keep hourly data for later filtering
        hourly_data = df.copy()
        
        # Also create daily aggregates (for non-0-day forecasts)
        daily_data = df.groupby('date').agg({
            'temperature_f': ['max', 'min', 'mean']
        }).reset_index()
        
        daily_data.columns = ['date', 'actual_high', 'actual_low', 'actual_average']
        
        # Find peak time
        peak_times = df.loc[df.groupby('date')['temperature_f'].idxmax()][['date', 'timestamp']]
        peak_times['peak_time'] = peak_times['timestamp'].dt.strftime('%H:%M')
        daily_data = daily_data.merge(peak_times[['date', 'peak_time']], on='date', how='left')
        
        logger.info(f"Loaded {len(daily_data)} days of actual temperatures")
        return daily_data, hourly_data
        
    except FileNotFoundError:
        logger.error(f"Actual temperature file not found: {filepath}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading actual temperatures: {e}")
        return pd.DataFrame(), pd.DataFrame()


def calculate_actual_high_from_time(hourly_data, target_date, from_time):
    """
    Calculate the actual high temperature from a specific time onwards on a given date.
    
    Args:
        hourly_data: DataFrame with hourly temperature data
        target_date: The date to analyze
        from_time: The datetime to start from (e.g., forecast issue time)
    
    Returns:
        max temperature from that time onwards, or None if no data
    """
    filtered = hourly_data[
        (hourly_data['date'] == target_date) & 
        (hourly_data['timestamp'] >= from_time)
    ]
    
    if len(filtered) > 0:
        return filtered['temperature_f'].max()
    return None


def merge_with_enhanced_0day(forecasts_df, daily_actuals, hourly_actuals):
    """
    Merge forecasts with actuals, handling 0-day forecasts specially.
    
    For 0-day forecasts: Compare forecasted_high (from forecast time onwards) 
                         with actual_high (from forecast time onwards)
    For 1+ day forecasts: Compare forecasted_high with full day actual_high
    """
    logger.info("Merging forecasts with actuals (enhanced 0-day handling)...")
    
    # Merge with daily actuals first
    merged = forecasts_df.merge(
        daily_actuals,
        left_on='target_date',
        right_on='date',
        how='left'
    )
    
    # Calculate lead time
    merged['lead_time_days'] = (merged['target_date'] - merged['forecast_date']).dt.days
    
    # For 0-day forecasts, recalculate actual_high from forecast time onwards
    logger.info("Recalculating actual_high for 0-day forecasts (from forecast time onwards)...")
    
    zero_day_mask = merged['lead_time_days'] == 0
    zero_day_count = zero_day_mask.sum()
    
    if zero_day_count > 0:
        logger.info(f"Processing {zero_day_count} 0-day forecasts...")
        
        # For each 0-day forecast, calculate actual high from forecast time onwards
        for idx in merged[zero_day_mask].index:
            target_date = merged.loc[idx, 'target_date']
            forecast_time = merged.loc[idx, 'forecast_datetime']
            
            actual_high_from_time = calculate_actual_high_from_time(
                hourly_actuals, target_date, forecast_time
            )
            
            if actual_high_from_time is not None:
                merged.loc[idx, 'actual_high'] = actual_high_from_time
    
    logger.info(f"Merged {len(merged)} forecast-actual pairs")
    return merged


def load_polymarket_odds(filepath='data/raw/polymarket_odds_history.csv'):
    """Load Polymarket historical odds data."""
    logger.info(f"Loading Polymarket odds from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        df['event_date'] = pd.to_datetime(df['event_date'])
        df['fetch_timestamp'] = pd.to_datetime(df['fetch_timestamp'])
        
        # Group by event_date and aggregate thresholds/probabilities into lists
        odds_grouped = df.groupby('event_date').agg({
            'threshold': lambda x: list(x),
            'threshold_type': lambda x: list(x),
            'yes_probability': lambda x: list(x),
            'volume': lambda x: list(x),
            'fetch_timestamp': 'first'
        }).reset_index()
        
        logger.info(f"Loaded odds for {len(odds_grouped)} event dates")
        return odds_grouped
        
    except FileNotFoundError:
        logger.warning(f"Polymarket odds file not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading Polymarket odds: {e}")
        return pd.DataFrame()


def merge_all_data(forecasts_df, daily_actuals, hourly_actuals, odds_df):
    """Merge forecasts, actuals, and odds into a single dataset."""
    
    # First merge forecasts with actuals (with enhanced 0-day handling)
    merged = merge_with_enhanced_0day(forecasts_df, daily_actuals, hourly_actuals)
    
    # Then merge with odds
    if not odds_df.empty:
        logger.info("Merging with Polymarket odds...")
        merged = merged.merge(
            odds_df,
            left_on='target_date',
            right_on='event_date',
            how='left'
        )
        logger.info(f"Final merged dataset: {len(merged)} rows")
    
    return merged


def main():
    """Main pipeline to merge all backtest data with enhanced 0-day handling."""
    
    logger.info("="*70)
    logger.info("ENHANCED BACKTEST DATA MERGER (v3)")
    logger.info("="*70)
    
    # Load data
    forecasts_df = load_historical_forecasts_enhanced()
    daily_actuals, hourly_actuals = load_actual_temperatures_enhanced()
    odds_df = load_polymarket_odds()
    
    if forecasts_df.empty or daily_actuals.empty:
        logger.error("Failed to load required data. Exiting.")
        return
    
    # Merge all data
    merged_df = merge_all_data(forecasts_df, daily_actuals, hourly_actuals, odds_df)
    
    # Save to CSV
    output_path = 'data/processed/backtest_data_enhanced.csv'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving merged data to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    
    logger.info("="*70)
    logger.info("MERGE COMPLETE")
    logger.info("="*70)
    logger.info(f"Output: {output_path}")
    logger.info(f"Total rows: {len(merged_df)}")
    logger.info(f"Date range: {merged_df['target_date'].min()} to {merged_df['target_date'].max()}")
    
    # Show sample statistics
    logger.info("\nSample statistics by lead time:")
    for lead_time in sorted(merged_df['lead_time_days'].unique()):
        subset = merged_df[merged_df['lead_time_days'] == lead_time]
        if len(subset) > 0:
            error = (subset['forecasted_high'] - subset['actual_high']).dropna()
            if len(error) > 0:
                mae = error.abs().mean()
                logger.info(f"  {int(lead_time)}-day: n={len(subset)}, MAE={mae:.2f}°F")


if __name__ == '__main__':
    main()
