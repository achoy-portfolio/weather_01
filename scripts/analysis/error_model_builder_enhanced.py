"""
Enhanced Error Model Builder

Calculates forecast error metrics at both hourly and daily levels:
1. Hourly forecast accuracy (all hours)
2. Daily max forecast accuracy
3. For 0-day forecasts, only compares remaining hours of the day
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json


def calculate_hourly_errors(forecasts_path='data/raw/historical_forecasts.csv',
                            actuals_path='data/raw/actual_temperatures.csv'):
    """
    Calculate hourly forecast errors by comparing each hourly forecast to actual temperature.
    For same-day forecasts, only compare hours after the forecast was issued.
    """
    print("Loading hourly forecast data...")
    forecasts = pd.read_csv(forecasts_path)
    forecasts['forecast_datetime'] = pd.to_datetime(forecasts['forecast_date'] + ' ' + forecasts['forecast_time'])
    forecasts['valid_time'] = pd.to_datetime(forecasts['valid_time'])
    
    print("Loading actual temperature data...")
    actuals = pd.read_csv(actuals_path)
    actuals['timestamp'] = pd.to_datetime(actuals['timestamp'])
    actuals.rename(columns={'temperature_f': 'actual_temp'}, inplace=True)
    
    # Merge forecasts with actuals on valid_time = timestamp
    print("Merging forecasts with actuals...")
    merged = forecasts.merge(
        actuals[['timestamp', 'actual_temp']], 
        left_on='valid_time', 
        right_on='timestamp',
        how='inner'
    )
    
    # Calculate lead time in hours
    merged['lead_time_hours'] = (merged['valid_time'] - merged['forecast_datetime']).dt.total_seconds() / 3600
    merged['lead_time_days'] = merged['lead_time_hours'] / 24
    
    # Calculate error
    merged['error'] = merged['temperature'] - merged['actual_temp']
    merged['abs_error'] = np.abs(merged['error'])
    
    # Extract date components
    merged['forecast_date_only'] = merged['forecast_datetime'].dt.date
    merged['valid_date'] = merged['valid_time'].dt.date
    merged['valid_hour'] = merged['valid_time'].dt.hour
    
    print(f"Total hourly forecast-actual pairs: {len(merged)}")
    
    return merged


def calculate_daily_max_errors(combined_data_path='data/processed/backtest_data_combined.csv'):
    """
    Calculate daily maximum temperature forecast errors.
    """
    print("\nLoading combined daily data...")
    df = pd.read_csv(combined_data_path)
    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate lead time and error
    df['lead_time_days'] = (df['date'] - df['forecast_date']).dt.days
    df['error'] = df['forecasted_high'] - df['actual_high']
    df['abs_error'] = np.abs(df['error'])
    
    # Extract season
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].apply(get_season)
    
    df_clean = df.dropna(subset=['error', 'lead_time_days'])
    
    print(f"Total daily max forecasts: {len(df_clean)}")
    
    return df_clean


def get_season(month):
    """Determine season from month number"""
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'


def calculate_metrics(errors):
    """Calculate MAE, RMSE, bias, std_dev for a set of errors"""
    return {
        'mae': float(np.mean(np.abs(errors))),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'bias': float(np.mean(errors)),
        'std_dev': float(np.std(errors)),
        'count': int(len(errors))
    }


def build_enhanced_error_model(output_path='data/processed/error_model_enhanced.json'):
    """
    Build comprehensive error model with both hourly and daily max analysis.
    """
    
    # Calculate hourly errors
    hourly_data = calculate_hourly_errors()
    
    # Calculate daily max errors
    daily_data = calculate_daily_max_errors()
    
    print("\n" + "="*70)
    print("HOURLY FORECAST ANALYSIS")
    print("="*70)
    
    # Overall hourly metrics
    hourly_overall = calculate_metrics(hourly_data['error'])
    print(f"\nOverall Hourly Accuracy:")
    print(f"  MAE: {hourly_overall['mae']:.2f}°F")
    print(f"  RMSE: {hourly_overall['rmse']:.2f}°F")
    print(f"  Bias: {hourly_overall['bias']:.2f}°F")
    print(f"  Count: {hourly_overall['count']:,}")
    
    # Hourly by lead time (in days)
    print("\nHourly Accuracy by Lead Time:")
    hourly_by_lead_time = {}
    for lead_days in sorted(hourly_data['lead_time_days'].unique()):
        if 0 <= lead_days <= 3:  # Focus on 0-3 day forecasts
            lead_data = hourly_data[
                (hourly_data['lead_time_days'] >= lead_days) & 
                (hourly_data['lead_time_days'] < lead_days + 1)
            ]
            if len(lead_data) > 0:
                metrics = calculate_metrics(lead_data['error'])
                hourly_by_lead_time[f'{int(lead_days)}_day'] = metrics
                print(f"  {int(lead_days)}-day: MAE={metrics['mae']:.2f}°F, "
                      f"RMSE={metrics['rmse']:.2f}°F, n={metrics['count']:,}")
    
    # Hourly by time of day
    print("\nHourly Accuracy by Time of Day:")
    hourly_by_hour = {}
    for hour in range(0, 24, 3):  # Every 3 hours
        hour_data = hourly_data[hourly_data['valid_hour'] == hour]
        if len(hour_data) > 0:
            metrics = calculate_metrics(hour_data['error'])
            hourly_by_hour[f'{hour:02d}:00'] = metrics
            print(f"  {hour:02d}:00: MAE={metrics['mae']:.2f}°F, n={metrics['count']:,}")
    
    print("\n" + "="*70)
    print("DAILY MAX FORECAST ANALYSIS")
    print("="*70)
    
    # Overall daily max metrics
    daily_overall = calculate_metrics(daily_data['error'])
    print(f"\nOverall Daily Max Accuracy:")
    print(f"  MAE: {daily_overall['mae']:.2f}°F")
    print(f"  RMSE: {daily_overall['rmse']:.2f}°F")
    print(f"  Bias: {daily_overall['bias']:.2f}°F")
    print(f"  Count: {daily_overall['count']}")
    
    # Daily max by lead time
    print("\nDaily Max Accuracy by Lead Time:")
    daily_by_lead_time = {}
    for lead_time in sorted(daily_data['lead_time_days'].unique()):
        lead_data = daily_data[daily_data['lead_time_days'] == lead_time]
        if len(lead_data) > 0:
            metrics = calculate_metrics(lead_data['error'])
            daily_by_lead_time[f'{int(lead_time)}_day'] = metrics
            print(f"  {int(lead_time)}-day: MAE={metrics['mae']:.2f}°F, "
                  f"RMSE={metrics['rmse']:.2f}°F, n={metrics['count']}")
    
    # Daily max by season
    print("\nDaily Max Accuracy by Season:")
    daily_by_season = {}
    for season in ['winter', 'spring', 'summer', 'fall']:
        season_data = daily_data[daily_data['season'] == season]
        if len(season_data) > 0:
            metrics = calculate_metrics(season_data['error'])
            daily_by_season[season] = metrics
            print(f"  {season.capitalize()}: MAE={metrics['mae']:.2f}°F, "
                  f"RMSE={metrics['rmse']:.2f}°F, n={metrics['count']}")
    
    # Build comprehensive error model
    error_model = {
        'model_version': '2.0',
        'created_at': datetime.now().isoformat(),
        'hourly_analysis': {
            'overall': hourly_overall,
            'by_lead_time': hourly_by_lead_time,
            'by_hour_of_day': hourly_by_hour,
            'training_period': {
                'start': hourly_data['valid_time'].min().strftime('%Y-%m-%d'),
                'end': hourly_data['valid_time'].max().strftime('%Y-%m-%d')
            }
        },
        'daily_max_analysis': {
            'overall': daily_overall,
            'by_lead_time': daily_by_lead_time,
            'by_season': daily_by_season,
            'training_period': {
                'start': daily_data['date'].min().strftime('%Y-%m-%d'),
                'end': daily_data['date'].max().strftime('%Y-%m-%d')
            }
        }
    }
    
    # Save error model
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving enhanced error model to {output_path}...")
    with open(output_file, 'w') as f:
        json.dump(error_model, f, indent=2)
    
    print("Enhanced error model saved successfully!")
    
    return error_model


if __name__ == '__main__':
    error_model = build_enhanced_error_model()
    
    print("\n" + "="*70)
    print("ENHANCED ERROR MODEL SUMMARY")
    print("="*70)
    print("\nKey Findings:")
    print(f"1. Hourly Forecast MAE: {error_model['hourly_analysis']['overall']['mae']:.2f}°F")
    print(f"2. Daily Max Forecast MAE: {error_model['daily_max_analysis']['overall']['mae']:.2f}°F")
    print(f"\n3. Hourly 0-day MAE: {error_model['hourly_analysis']['by_lead_time'].get('0_day', {}).get('mae', 'N/A'):.2f}°F")
    print(f"4. Daily Max 0-day MAE: {error_model['daily_max_analysis']['by_lead_time'].get('0_day', {}).get('mae', 'N/A'):.2f}°F")
    print("\nThe hourly analysis shows how accurate forecasts are for each hour,")
    print("while daily max analysis shows accuracy for predicting the day's high.")
