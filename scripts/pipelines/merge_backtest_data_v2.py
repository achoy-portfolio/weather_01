"""
Data Merger and Validator for Backtest Pipeline (Version 2)

This script merges historical forecasts, actual temperatures, and Polymarket odds
into a single dataset for backtesting. It correctly matches odds based on forecast timing.

Key Fix: Odds are matched based on forecast_date at 9 PM, not just event_date.

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
FORECAST_HOUR = 21  # 9 PM Eastern Time


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


def load_polymarket_odds(filepath='data/raw/polymarket_odds_history.csv'):
    """
    Load Polymarket odds data with full timestamp information.
    
    Returns DataFrame with all odds records including fetch_timestamp.
    """
    logger.info(f"Loading Polymarket odds from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        # Parse dates
        df['event_date'] = pd.to_datetime(df['event_date'])
        
        if 'fetch_timestamp' in df.columns:
            df['fetch_timestamp'] = pd.to_datetime(df['fetch_timestamp'], utc=True)
        
        logger.info(f"Loaded {len(df)} odds records for {df['event_date'].nunique()} unique dates")
        return df
        
    except FileNotFoundError:
        logger.error(f"Polymarket odds file not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading Polymarket odds: {e}")
        return pd.DataFrame()


def match_odds_to_forecast(forecast_row, odds_df, forecast_hour=21):
    """
    Match odds to a specific forecast based on timing.
    
    Per Requirements 1.2 and 4.1: Use odds available at forecast_date 9 PM
    for the target event_date.
    
    Args:
        forecast_row: Single row from forecasts DataFrame
        odds_df: Full odds DataFrame
        forecast_hour: Hour when forecast is made (default 21 for 9 PM)
    
    Returns: Dictionary with odds data or None if no match
    """
    forecast_date = forecast_row['forecast_date']
    target_date = forecast_row['target_date']
    
    # Get all odds for this event_date
    event_odds = odds_df[odds_df['event_date'] == target_date]
    
    if event_odds.empty:
        return None
    
    # Target time: forecast_date at 9 PM Eastern
    # Convert to UTC (Eastern is UTC-5 in winter, UTC-4 in summer)
    # For simplicity, using UTC-5 (EST)
    target_time = pd.Timestamp(forecast_date).replace(hour=forecast_hour, minute=0, second=0)
    target_time_utc = pd.Timestamp(target_time).tz_localize('US/Eastern').tz_convert('UTC')
    
    # Find odds fetch closest to (but not after) the forecast time
    # We want odds that were available WHEN the forecast was made
    event_odds = event_odds.copy()
    event_odds['time_diff'] = (event_odds['fetch_timestamp'] - target_time_utc).dt.total_seconds()
    
    # Filter to odds available before or at forecast time
    available_odds = event_odds[event_odds['time_diff'] <= 0]
    
    if available_odds.empty:
        # If no odds before forecast time, take the earliest available
        closest_odds = event_odds.loc[event_odds['time_diff'].abs().idxmin()]
        fetch_time = closest_odds['fetch_timestamp']
    else:
        # Take the most recent odds before forecast time
        closest_odds = available_odds.loc[available_odds['time_diff'].abs().idxmin()]
        fetch_time = closest_odds['fetch_timestamp']
    
    # Get all thresholds from this fetch
    fetch_odds = event_odds[event_odds['fetch_timestamp'] == fetch_time]
    
    return {
        'threshold': list(fetch_odds['threshold']),
        'threshold_type': list(fetch_odds['threshold_type']),
        'yes_probability': list(fetch_odds['yes_probability']),
        'volume': list(fetch_odds['volume']),
        'fetch_timestamp': fetch_time
    }


def merge_datasets(forecasts_df, actuals_df, odds_df):
    """
    Merge forecast, actual, and odds data with correct timing logic.
    
    Key: Odds are matched based on forecast_date at 9 PM, not just event_date.
    
    Returns merged DataFrame with all data aligned by forecast and target date.
    """
    logger.info("Merging datasets with time-aware odds matching...")
    
    if forecasts_df.empty:
        logger.error("No forecast data to merge")
        return pd.DataFrame()
    
    # Start with forecasts
    merged = forecasts_df.copy()
    merged.rename(columns={'target_date': 'date'}, inplace=True)
    
    # Merge with actuals (simple join on date)
    if not actuals_df.empty:
        merged = merged.merge(actuals_df, on='date', how='left')
        logger.info(f"After merging actuals: {len(merged)} records")
    
    # Merge with odds (complex time-based matching)
    if not odds_df.empty:
        logger.info("Matching odds to forecasts based on timing...")
        
        odds_data = []
        for idx, row in merged.iterrows():
            odds_match = match_odds_to_forecast(row, odds_df)
            odds_data.append(odds_match)
        
        # Add odds columns to merged dataframe
        if odds_data:
            odds_df_matched = pd.DataFrame(odds_data)
            merged = pd.concat([merged, odds_df_matched], axis=1)
        
        logger.info(f"After merging odds: {len(merged)} records")
    
    # Sort by forecast_date and date
    merged = merged.sort_values(['date', 'forecast_date']).reset_index(drop=True)
    
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
            logger.warning(f"Missing forecasts for {missing_forecasts} records")
    
    # Check for missing actuals
    if 'actual_high' in merged_df.columns:
        missing_actuals = merged_df['actual_high'].isna().sum()
        validation_results['missing_actuals'] = missing_actuals
        if missing_actuals > 0:
            logger.warning(f"Missing actual temperatures for {missing_actuals} records")
    
    # Check for missing odds
    if 'threshold' in merged_df.columns:
        missing_odds = merged_df['threshold'].isna().sum()
        validation_results['missing_odds'] = missing_odds
        if missing_odds > 0:
            logger.warning(f"Missing odds data for {missing_odds} records")
    
    # Count complete records
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
    
    # Get unique dates for rolling average calculation
    daily_temps = merged_df.groupby('date')['actual_high'].first().reset_index()
    
    # Calculate rolling average
    daily_temps['rolling_avg'] = daily_temps['actual_high'].rolling(
        window=window_days, 
        min_periods=1
    ).mean()
    
    # Detect outliers
    daily_temps['temp_deviation'] = abs(daily_temps['actual_high'] - daily_temps['rolling_avg'])
    outlier_mask = daily_temps['temp_deviation'] > OUTLIER_THRESHOLD
    
    if outlier_mask.any():
        outlier_records = daily_temps[outlier_mask]
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
    logger.info("Starting Data Merger and Validator (V2 - Time-Aware)")
    logger.info("=" * 60)
    
    # Load data
    forecasts_df = load_historical_forecasts()
    actuals_df = load_actual_temperatures()
    odds_df = load_polymarket_odds()
    
    # Merge datasets with time-aware odds matching
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
