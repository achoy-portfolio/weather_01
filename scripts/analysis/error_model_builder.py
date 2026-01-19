"""
Error Model Builder

Calculates forecast error metrics including MAE, RMSE, bias, and standard deviation.
Breaks down errors by lead time and season.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json


def calculate_forecast_error(forecast_temp, actual_temp):
    """Calculate error for a single forecast (forecast - actual)"""
    return forecast_temp - actual_temp


def calculate_mae(errors):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(errors))


def calculate_rmse(errors):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(np.mean(errors ** 2))


def calculate_bias(errors):
    """Calculate bias (mean error) - detects systematic over/under prediction"""
    return np.mean(errors)


def calculate_std_dev(errors):
    """Calculate standard deviation of errors"""
    return np.std(errors)


def get_season(month):
    """Determine season from month number"""
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:  # 9, 10, 11
        return 'fall'


def calculate_lead_time(forecast_date, target_date):
    """Calculate lead time in days between forecast and target date"""
    if isinstance(forecast_date, str):
        forecast_date = pd.to_datetime(forecast_date)
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    return (target_date - forecast_date).days


def build_error_model(combined_data_path='data/processed/backtest_data_combined.csv',
                      output_path='data/processed/error_model.json'):
    """
    Build error model from combined backtest data.
    
    Calculates:
    - Overall MAE, RMSE, bias, std_dev
    - Error metrics by lead time (1-day, 2-day, etc.)
    - Error metrics by season (winter, spring, summer, fall)
    """
    
    print("Loading combined backtest data...")
    df = pd.read_csv(combined_data_path)
    
    # Convert date columns to datetime
    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate forecast errors
    print("Calculating forecast errors...")
    df['forecast_error'] = df['forecasted_high'] - df['actual_high']
    
    # Calculate lead time
    df['lead_time_days'] = df.apply(
        lambda row: calculate_lead_time(row['forecast_date'], row['date']),
        axis=1
    )
    
    # Extract month and season from target date
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].apply(get_season)
    
    # Remove rows with missing data
    df_clean = df.dropna(subset=['forecast_error', 'lead_time_days'])
    
    print(f"Total forecasts analyzed: {len(df_clean)}")
    
    # Calculate overall metrics
    print("\nCalculating overall error metrics...")
    overall_metrics = {
        'mae': float(calculate_mae(df_clean['forecast_error'])),
        'rmse': float(calculate_rmse(df_clean['forecast_error'])),
        'bias': float(calculate_bias(df_clean['forecast_error'])),
        'std_dev': float(calculate_std_dev(df_clean['forecast_error']))
    }
    
    print(f"  MAE: {overall_metrics['mae']:.2f}°F")
    print(f"  RMSE: {overall_metrics['rmse']:.2f}°F")
    print(f"  Bias: {overall_metrics['bias']:.2f}°F")
    print(f"  Std Dev: {overall_metrics['std_dev']:.2f}°F")
    
    # Calculate metrics by lead time
    print("\nCalculating error metrics by lead time...")
    by_lead_time = {}
    for lead_time in sorted(df_clean['lead_time_days'].unique()):
        lead_data = df_clean[df_clean['lead_time_days'] == lead_time]
        if len(lead_data) > 0:
            errors = lead_data['forecast_error']
            by_lead_time[f'{int(lead_time)}_day'] = {
                'mae': float(calculate_mae(errors)),
                'rmse': float(calculate_rmse(errors)),
                'bias': float(calculate_bias(errors)),
                'std_dev': float(calculate_std_dev(errors)),
                'count': int(len(lead_data))
            }
            print(f"  {int(lead_time)}-day: MAE={by_lead_time[f'{int(lead_time)}_day']['mae']:.2f}°F, "
                  f"RMSE={by_lead_time[f'{int(lead_time)}_day']['rmse']:.2f}°F, "
                  f"n={by_lead_time[f'{int(lead_time)}_day']['count']}")
    
    # Calculate metrics by season
    print("\nCalculating error metrics by season...")
    by_season = {}
    for season in ['winter', 'spring', 'summer', 'fall']:
        season_data = df_clean[df_clean['season'] == season]
        if len(season_data) > 0:
            errors = season_data['forecast_error']
            by_season[season] = {
                'mae': float(calculate_mae(errors)),
                'rmse': float(calculate_rmse(errors)),
                'bias': float(calculate_bias(errors)),
                'std_dev': float(calculate_std_dev(errors)),
                'count': int(len(season_data))
            }
            print(f"  {season.capitalize()}: MAE={by_season[season]['mae']:.2f}°F, "
                  f"RMSE={by_season[season]['rmse']:.2f}°F, "
                  f"n={by_season[season]['count']}")
    
    # Build error model structure
    error_model = {
        'model_version': '1.0',
        'created_at': datetime.now().isoformat(),
        'training_period': {
            'start_date': df_clean['date'].min().strftime('%Y-%m-%d'),
            'end_date': df_clean['date'].max().strftime('%Y-%m-%d'),
            'num_forecasts': int(len(df_clean))
        },
        'overall_metrics': overall_metrics,
        'by_lead_time': by_lead_time,
        'by_season': by_season
    }
    
    # Save error model to JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving error model to {output_path}...")
    with open(output_file, 'w') as f:
        json.dump(error_model, f, indent=2)
    
    print("Error model saved successfully!")
    
    return error_model


if __name__ == '__main__':
    error_model = build_error_model()
    
    print("\n" + "="*60)
    print("ERROR MODEL SUMMARY")
    print("="*60)
    print(f"\nTraining Period: {error_model['training_period']['start_date']} to "
          f"{error_model['training_period']['end_date']}")
    print(f"Total Forecasts: {error_model['training_period']['num_forecasts']}")
    print(f"\nOverall Performance:")
    print(f"  MAE:     {error_model['overall_metrics']['mae']:.2f}°F")
    print(f"  RMSE:    {error_model['overall_metrics']['rmse']:.2f}°F")
    print(f"  Bias:    {error_model['overall_metrics']['bias']:.2f}°F")
    print(f"  Std Dev: {error_model['overall_metrics']['std_dev']:.2f}°F")
