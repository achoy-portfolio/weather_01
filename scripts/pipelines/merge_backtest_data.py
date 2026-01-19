"""
Data Merger and Validator for Backtest Pipeline

This script merges historical forecasts, actual temperatures, and Polymarket odds
into a single dataset for backtesting. It also validates data quality and completeness.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
NYC_TEMP_MIN = -20.0  # Minimum reasonable temperature for NYC (°F)
NYC_TEMP_MAX = 120.0  # Maximum reasonable temperature for NYC (°F)
OUTLIER_THRESHOLD = 20.0  # Flag temps that differ by more than this from recent average


def load_historical_forecasts(filepath='data/raw/historical_forecasts.csv'):
    """
    Load and aggregate historical forecasts to daily peak forecasts.
    
    Returns DataFrame with columns: forecast_date, target_date, forecasted_high
    """
    logger.info(f"Loading historical forecasts from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        # Parse dates
        df['forecast_date'] = pd.to_datetime(df['forecast_date'])
        df['valid_time'] = pd.to_datetime(df['valid_time'])
        
        # Extract target date from valid_time
        df['target_date'] = df['valid_time'].dt.date
        
        # Group by forecast_date and target_date, get max temperature (peak forecast)
        daily_forecasts = df.groupby(['forecast_date', 'target_date']).agg({
            'temperature': 'max',
            'source': 'first'
        }).reset_index()
        
        daily_forecasts.rename(columns={'temperature': 'forecasted_high'}, inplace=True)
        daily_forecasts['target_date'] = pd.to_datetime(daily_forecasts['target_date'])
        
        logger.info(f"Loaded {len(daily_forecasts)} daily forecasts")
        return daily_forecasts
        
    except FileNotFoundError:
        logger.error(f"Forecast file not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading forecasts: {e}")
        return pd.DataFrame()


def load_actual_temperatures(filepath='data/raw/actual_temperatures.csv'):
    """
    Load and aggregate actual temperatures to daily highs, lows, and averages.
    
    Returns DataFrame with columns: date, actual_high, actual_low, actual_average, peak_time
    """
    logger.info(f"Loading actual temperatures from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Group by date and calculate daily statistics
        daily_temps = df.groupby('date').agg({
            'temperature_f': ['max', 'min', 'mean']
        }).reset_index()
        
        # Flatten column names
        daily_temps.columns = ['date', 'actual_high', 'actual_low', 'actual_average']
        
        # Find peak time for each day
        peak_times = df.loc[df.groupby('date')['temperature_f'].idxmax()][['date', 'timestamp']]
        peak_times['peak_time'] = peak_times['timestamp'].dt.strftime('%H:%M')
        
        # Merge peak times
        daily_temps = daily_temps.merge(peak_times[['date', 'peak_time']], on='date', how='left')
        daily_temps['date'] = pd.to_datetime(daily_temps['date'])
        
        logger.info(f"Loaded {len(daily_temps)} days of actual temperatures")
        return daily_temps
        
    except FileNotFoundError:
        logger.error(f"Actual temperature file not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading actual temperatures: {e}")
        return pd.DataFrame()


def load_polymarket_odds(filepath='data/raw/polymarket_odds_history.csv', target_hour=21):
    """
    Load Polymarket odds data and filter to 9 PM the day before each event.
    
    Per Requirements 1.2 and 4.1: Use odds from 9 PM Eastern Time the day before
    the target date, which is when forecasts are made and betting decisions occur.
    
    Args:
        filepath: Path to odds CSV file
        target_hour: Hour to filter odds (default 21 for 9 PM Eastern)
    
    Returns DataFrame with columns: event_date, threshold, threshold_type, yes_probability, volume, fetch_timestamp
    """
    logger.info(f"Loading Polymarket odds from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        # Parse event_date
        df['event_date'] = pd.to_datetime(df['event_date'])
        
        # Parse fetch_timestamp if present
        if 'fetch_timestamp' in df.columns:
            df['fetch_timestamp'] = pd.to_datetime(df['fetch_timestamp'], utc=True)
            
            # For each event_date, find the odds closest to 9 PM the day before
            # This is when betting decisions are made based on the 9 PM forecast
            filtered_odds = []
            
            for event_date in df['event_date'].unique():
                event_odds = df[df['event_date'] == event_date].copy()
                
                # Target time: 9 PM Eastern the day before the event
                # Convert to UTC for comparison (Eastern is UTC-5 or UTC-4 depending on DST)
                target_time_local = pd.Timestamp(event_date) - pd.Timedelta(days=1)
                target_time_local = target_time_local.replace(hour=target_hour, minute=0, second=0)
                
                # Convert to UTC (assuming EST = UTC-5 for winter months)
                target_time_utc = pd.Timestamp(target_time_local, tz='UTC') + pd.Timedelta(hours=5)
                
                # Find the fetch closest to target time (before or at the target time)
                # We want odds available AT decision time, not after
                before_target = event_odds[event_odds['fetch_timestamp'] <= target_time_utc]
                
                if len(before_target) > 0:
                    # Use the most recent odds before/at 9 PM
                    closest_fetch = before_target['fetch_timestamp'].max()
                else:
                    # If no odds before target, use the earliest available
                    closest_fetch = event_odds['fetch_timestamp'].min()
                    logger.warning(f"No odds before 9 PM for {event_date}, using earliest available: {closest_fetch}")
                
                # Keep all thresholds from that fetch
                closest_odds = event_odds[event_odds['fetch_timestamp'] == closest_fetch]
                filtered_odds.append(closest_odds)
            
            df = pd.concat(filtered_odds, ignore_index=True)
            logger.info(f"Filtered to odds at/before {target_hour}:00 (9 PM) the day before each event")
        
        logger.info(f"Loaded {len(df)} odds records for {df['event_date'].nunique()} unique dates")
        return df
        
    except FileNotFoundError:
        logger.error(f"Polymarket odds file not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading Polymarket odds: {e}")
        return pd.DataFrame()


def merge_datasets(forecasts_df, actuals_df, odds_df):
    """
    Merge forecast, actual, and odds data by date.
    
    Returns merged DataFrame with all data aligned by date.
    """
    logger.info("Merging datasets...")
    
    # Start with forecasts (using target_date as the key)
    if forecasts_df.empty:
        logger.warning("No forecast data to merge")
        merged = pd.DataFrame()
    else:
        merged = forecasts_df.copy()
        merged.rename(columns={'target_date': 'date'}, inplace=True)
    
    # Merge with actuals
    if not actuals_df.empty and not merged.empty:
        merged = merged.merge(actuals_df, on='date', how='outer', indicator='_merge_actuals')
        logger.info(f"After merging actuals: {len(merged)} records")
    elif not actuals_df.empty:
        merged = actuals_df.copy()
        merged['_merge_actuals'] = 'right_only'
    
    # Merge with odds (odds are keyed by event_date)
    if not odds_df.empty and not merged.empty:
        # For odds, we'll keep them in a nested structure or separate
        # Since one date can have multiple thresholds, we'll create a summary
        odds_summary = odds_df.groupby('event_date').agg({
            'threshold': lambda x: list(x),
            'threshold_type': lambda x: list(x),
            'yes_probability': lambda x: list(x),
            'volume': lambda x: list(x),
            'fetch_timestamp': 'first'  # Keep the fetch timestamp for verification
        }).reset_index()
        odds_summary.rename(columns={'event_date': 'date'}, inplace=True)
        
        merged = merged.merge(odds_summary, on='date', how='outer', indicator='_merge_odds')
        logger.info(f"After merging odds: {len(merged)} records")
    
    # Sort by date
    if not merged.empty and 'date' in merged.columns:
        merged = merged.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Final merged dataset: {len(merged)} records")
    return merged


def validate_data_completeness(merged_df):
    """
    Validate data completeness and log missing data.
    
    Requirement 7.2: Check for missing forecasts, actuals, or odds
    """
    logger.info("Validating data completeness...")
    
    validation_results = {
        'total_records': len(merged_df),
        'missing_forecasts': 0,
        'missing_actuals': 0,
        'missing_odds': 0,
        'complete_records': 0
    }
    
    if merged_df.empty:
        logger.warning("No data to validate")
        return validation_results
    
    # Check for missing forecasts
    if 'forecasted_high' in merged_df.columns:
        missing_forecasts = merged_df['forecasted_high'].isna().sum()
        validation_results['missing_forecasts'] = missing_forecasts
        if missing_forecasts > 0:
            logger.warning(f"Missing forecasts for {missing_forecasts} dates")
            missing_dates = merged_df[merged_df['forecasted_high'].isna()]['date'].tolist()
            logger.debug(f"Dates with missing forecasts: {missing_dates[:5]}...")
    
    # Check for missing actuals
    if 'actual_high' in merged_df.columns:
        missing_actuals = merged_df['actual_high'].isna().sum()
        validation_results['missing_actuals'] = missing_actuals
        if missing_actuals > 0:
            logger.warning(f"Missing actual temperatures for {missing_actuals} dates")
            missing_dates = merged_df[merged_df['actual_high'].isna()]['date'].tolist()
            logger.debug(f"Dates with missing actuals: {missing_dates[:5]}...")
    
    # Check for missing odds
    if 'threshold' in merged_df.columns:
        missing_odds = merged_df['threshold'].isna().sum()
        validation_results['missing_odds'] = missing_odds
        if missing_odds > 0:
            logger.warning(f"Missing odds data for {missing_odds} dates")
            missing_dates = merged_df[merged_df['threshold'].isna()]['date'].tolist()
            logger.debug(f"Dates with missing odds: {missing_dates[:5]}...")
    
    # Count complete records (have all three data types)
    if all(col in merged_df.columns for col in ['forecasted_high', 'actual_high', 'threshold']):
        complete_mask = (
            merged_df['forecasted_high'].notna() &
            merged_df['actual_high'].notna() &
            merged_df['threshold'].notna()
        )
        validation_results['complete_records'] = complete_mask.sum()
        logger.info(f"Complete records (all data present): {validation_results['complete_records']}")
    
    return validation_results


def detect_outliers(merged_df, window_days=7):
    """
    Detect temperature outliers based on recent average.
    
    Requirement 7.3: Flag values >20°F from recent average
    """
    logger.info(f"Detecting outliers (threshold: {OUTLIER_THRESHOLD}°F from {window_days}-day average)...")
    
    outliers = []
    
    if 'actual_high' not in merged_df.columns:
        logger.warning("No actual temperature data to check for outliers")
        return outliers
    
    # Calculate rolling average
    merged_df['rolling_avg'] = merged_df['actual_high'].rolling(
        window=window_days, 
        min_periods=1
    ).mean()
    
    # Detect outliers in actual temperatures
    merged_df['temp_deviation'] = abs(merged_df['actual_high'] - merged_df['rolling_avg'])
    outlier_mask = merged_df['temp_deviation'] > OUTLIER_THRESHOLD
    
    if outlier_mask.any():
        outlier_records = merged_df[outlier_mask][['date', 'actual_high', 'rolling_avg', 'temp_deviation']]
        for _, row in outlier_records.iterrows():
            outlier_info = {
                'date': row['date'],
                'actual_high': row['actual_high'],
                'rolling_avg': row['rolling_avg'],
                'deviation': row['temp_deviation']
            }
            outliers.append(outlier_info)
            logger.warning(
                f"Outlier detected on {row['date']}: "
                f"actual={row['actual_high']:.1f}°F, "
                f"avg={row['rolling_avg']:.1f}°F, "
                f"deviation={row['temp_deviation']:.1f}°F"
            )
    else:
        logger.info("No outliers detected")
    
    return outliers


def validate_temperature_bounds(merged_df):
    """
    Validate that temperatures are within reasonable bounds for NYC.
    
    Requirement 7.4: Validate temperature bounds (-20°F to 120°F for NYC)
    """
    logger.info(f"Validating temperature bounds ({NYC_TEMP_MIN}°F to {NYC_TEMP_MAX}°F)...")
    
    invalid_temps = []
    
    # Check forecasted temperatures
    if 'forecasted_high' in merged_df.columns:
        invalid_forecast = merged_df[
            (merged_df['forecasted_high'] < NYC_TEMP_MIN) | 
            (merged_df['forecasted_high'] > NYC_TEMP_MAX)
        ]
        if not invalid_forecast.empty:
            for _, row in invalid_forecast.iterrows():
                invalid_temps.append({
                    'date': row['date'],
                    'type': 'forecast',
                    'value': row['forecasted_high']
                })
                logger.error(
                    f"Invalid forecast temperature on {row['date']}: "
                    f"{row['forecasted_high']:.1f}°F (out of bounds)"
                )
    
    # Check actual temperatures
    for col in ['actual_high', 'actual_low', 'actual_average']:
        if col in merged_df.columns:
            invalid_actual = merged_df[
                (merged_df[col] < NYC_TEMP_MIN) | 
                (merged_df[col] > NYC_TEMP_MAX)
            ]
            if not invalid_actual.empty:
                for _, row in invalid_actual.iterrows():
                    invalid_temps.append({
                        'date': row['date'],
                        'type': col,
                        'value': row[col]
                    })
                    logger.error(
                        f"Invalid {col} on {row['date']}: "
                        f"{row[col]:.1f}°F (out of bounds)"
                    )
    
    if not invalid_temps:
        logger.info("All temperatures within valid bounds")
    
    return invalid_temps


def validate_consistency(merged_df):
    """
    Verify temperature consistency (high >= low, average between high and low).
    
    Requirement 7.5: Verify consistency
    """
    logger.info("Validating temperature consistency...")
    
    inconsistencies = []
    
    required_cols = ['actual_high', 'actual_low', 'actual_average']
    if not all(col in merged_df.columns for col in required_cols):
        logger.warning("Missing columns for consistency check")
        return inconsistencies
    
    # Check high >= low
    invalid_high_low = merged_df[merged_df['actual_high'] < merged_df['actual_low']]
    if not invalid_high_low.empty:
        for _, row in invalid_high_low.iterrows():
            inconsistencies.append({
                'date': row['date'],
                'issue': 'high < low',
                'high': row['actual_high'],
                'low': row['actual_low']
            })
            logger.error(
                f"Inconsistency on {row['date']}: "
                f"high ({row['actual_high']:.1f}°F) < low ({row['actual_low']:.1f}°F)"
            )
    
    # Check average between high and low
    invalid_avg = merged_df[
        (merged_df['actual_average'] < merged_df['actual_low']) |
        (merged_df['actual_average'] > merged_df['actual_high'])
    ]
    if not invalid_avg.empty:
        for _, row in invalid_avg.iterrows():
            inconsistencies.append({
                'date': row['date'],
                'issue': 'average out of range',
                'high': row['actual_high'],
                'low': row['actual_low'],
                'average': row['actual_average']
            })
            logger.error(
                f"Inconsistency on {row['date']}: "
                f"average ({row['actual_average']:.1f}°F) not between "
                f"low ({row['actual_low']:.1f}°F) and high ({row['actual_high']:.1f}°F)"
            )
    
    if not inconsistencies:
        logger.info("All temperature data is consistent")
    
    return inconsistencies


def save_merged_data(merged_df, output_path='data/processed/backtest_data_combined.csv'):
    """
    Save merged data to CSV file.
    
    Requirement 7.6: Save merged data
    """
    logger.info(f"Saving merged data to {output_path}")
    
    try:
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert list columns to string representation for CSV
        df_to_save = merged_df.copy()
        for col in ['threshold', 'threshold_type', 'yes_probability', 'volume']:
            if col in df_to_save.columns:
                df_to_save[col] = df_to_save[col].apply(
                    lambda x: str(x) if isinstance(x, list) else x
                )
        
        # Drop temporary merge indicator columns
        cols_to_drop = [col for col in df_to_save.columns if col.startswith('_merge')]
        if cols_to_drop:
            df_to_save = df_to_save.drop(columns=cols_to_drop)
        
        # Save to CSV
        df_to_save.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df_to_save)} records to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving merged data: {e}")
        return False


def main():
    """
    Main function to orchestrate data merging and validation.
    """
    logger.info("=" * 60)
    logger.info("Starting Data Merger and Validator")
    logger.info("=" * 60)
    
    # Load data
    forecasts_df = load_historical_forecasts()
    actuals_df = load_actual_temperatures()
    odds_df = load_polymarket_odds()
    
    # Merge datasets
    merged_df = merge_datasets(forecasts_df, actuals_df, odds_df)
    
    if merged_df.empty:
        logger.error("No data to process. Exiting.")
        return
    
    # Validate data
    logger.info("\n" + "=" * 60)
    logger.info("Data Validation")
    logger.info("=" * 60)
    
    completeness = validate_data_completeness(merged_df)
    outliers = detect_outliers(merged_df)
    invalid_bounds = validate_temperature_bounds(merged_df)
    inconsistencies = validate_consistency(merged_df)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)
    logger.info(f"Total records: {completeness['total_records']}")
    logger.info(f"Complete records: {completeness['complete_records']}")
    logger.info(f"Missing forecasts: {completeness['missing_forecasts']}")
    logger.info(f"Missing actuals: {completeness['missing_actuals']}")
    logger.info(f"Missing odds: {completeness['missing_odds']}")
    logger.info(f"Outliers detected: {len(outliers)}")
    logger.info(f"Invalid bounds: {len(invalid_bounds)}")
    logger.info(f"Inconsistencies: {len(inconsistencies)}")
    
    # Save merged data
    success = save_merged_data(merged_df)
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("Data merge and validation completed successfully!")
        logger.info("=" * 60)
    else:
        logger.error("Failed to save merged data")


if __name__ == "__main__":
    main()
